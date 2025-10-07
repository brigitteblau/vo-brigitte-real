import time, argparse, collections
import cv2
import numpy as np

from control.controller import PrintBackend  # En Raspi: cambia a GPIOBackend
from control.planner import SimplePlanner
from control.obstacle import detect_dummy_keyboard

def score_diff(prev, cur, roi=None):
    """Suma de diferencias en ROI (grayscale)."""
    if prev is None or cur is None:
        return 0.0
    a, b = prev, cur
    if roi is not None:
        x0,y0,x1,y1 = roi
        a = a[y0:y1, x0:x1]
        b = b[y0:y1, x0:x1]
    diff = cv2.absdiff(a, b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return float(gray.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Índice de cámara (por defecto 0)")
    ap.add_argument("--video", type=str, default="", help="Ruta a video en vez de cámara")
    ap.add_argument("--use_keyboard", action="store_true", help="Enter simula obstáculo")
    ap.add_argument("--show", action="store_true", help="Mostrar preview con ROI y score")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)

    # Detección
    ap.add_argument("--roi_frac", type=float, default=0.33, help="Fracción central para ROI (0.33 = 1/3)")
    ap.add_argument("--th_on", type=float, default=45_000, help="Umbral superior (se activa obstáculo)")
    ap.add_argument("--th_off", type=float, default=30_000, help="Umbral inferior (se desactiva obstáculo)")
    ap.add_argument("--win", type=int, default=5, help="Ventana para suavizar score")
    ap.add_argument("--cooldown", type=float, default=0.15, help="Tiempo mínimo entre evaluaciones (s)")

    # Planificación / comandos
    ap.add_argument("--turn_s", type=float, default=0.6, help="Duración giro izquierda (s)")
    ap.add_argument("--step_s", type=float, default=0.25, help="Duración paso adelante (s)")
    ap.add_argument("--min_dwell", type=float, default=0.35, help="Tiempo mínimo para cambiar comando (s)")

    args = ap.parse_args()

    # Backend
    backend = PrintBackend()  # >>> En Raspi: from control.controller import GPIOBackend; backend = GPIOBackend()
    planner = SimplePlanner(turn_seconds=args.turn_s, forward_chunk=args.step_s)

    # Fuente de frames
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("No pude abrir cámara/video.")
        return

    last_frame = None
    last_eval_t = 0.0
    obstacle = False  # estado con histeresis
    scores = collections.deque(maxlen=max(1, args.win))

    # Debounce de comandos
    last_cmd = None
    last_change_t = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Fin del stream o sin frame.")
                break

            now = time.time()
            dt = max(now - last_eval_t, 1e-6)

            # ROI central
            h, w = frame.shape[:2]
            rw = int(w * args.roi_frac)
            rh = int(h * args.roi_frac)
            x0 = (w - rw) // 2
            y0 = (h - rh) // 2
            roi = (x0, y0, x0 + rw, y0 + rh)

            # 1) Obtener señal de obstáculo
            if args.use_keyboard:
                obs = detect_dummy_keyboard()
                score = 0.0
            else:
                score = score_diff(last_frame, frame, roi=roi)
                scores.append(score)
                smooth = np.mean(scores)

                # Histeresis: solo prendemos si pasa th_on; solo apagamos si baja de th_off
                if not obstacle and smooth >= args.th_on:
                    obstacle = True
                elif obstacle and smooth <= args.th_off:
                    obstacle = False
                obs = obstacle

            # 2) Throttle de evaluación (cooldown)
            if now - last_eval_t >= args.cooldown:
                # 3) Planner -> lista de acciones
                actions = planner.step(obstacle_ahead=obs)
                # 4) Debounce por dwell: no cambiar hasta sostener un rato
                for cmd, dur in actions:
                    if cmd != last_cmd:
                        if (now - last_change_t) >= args.min_dwell or last_cmd is None:
                            backend.send(cmd, dur)
                            last_cmd = cmd
                            last_change_t = time.time()
                    else:
                        backend.send(cmd, dur)
                last_eval_t = now

            # 5) UI opcional
            if args.show:
                # dibujar ROI
                x0,y0,x1,y1 = roi
                cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2)
                # overlay de datos
                txt1 = f"score:{int(score)}  smooth:{int(np.mean(scores) if scores else 0)}  obs:{obs}"
                txt2 = f"cmd:{last_cmd or '-'}"
                cv2.putText(frame, txt1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                cv2.putText(frame, txt1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
                cv2.putText(frame, txt2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                cv2.putText(frame, txt2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
                cv2.imshow("run_cam", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
                    break

            last_frame = frame

    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        # En Raspi con GPIOBackend: backend.cleanup()
