# main_mt.py
import time, cv2, queue, yaml, numpy as np
from threading import Lock

from sensors.imu import IMU
from sensors.camera import open_camera
from vo.vo_hibrido import VOHibrido          
from estimator.estimator import Estimator2D
from filter.ekf import EKF2D
from control.pid import LaneController
from control.commander import Commander
from threads import IMUThread, CameraThread

def load_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # 0) Cargar config ANTES de usar cfg
    cfg = load_cfg()

    # 1) Cámara
    cam, K_auto, dist_auto = open_camera(
        cfg["camera"]["index"],
        cfg["camera"]["width"],
        cfg["camera"]["height"],
        cfg["camera"]["fps"],
    )

    # si tenés K/dist en YAML, úsalo; si no, lo auto
    if "K" in cfg["camera"] and "dist" in cfg["camera"]:
        K   = np.array(cfg["camera"]["K"], dtype=np.float32)
        dist= np.array(cfg["camera"]["dist"], dtype=np.float32)
    else:
        K, dist = K_auto, dist_auto

    # 2) Sensores y módulos
    imu = IMU()
    print("[IMU] Calibrando...")
    imu.calibrate(n=400)
    print("[IMU] OK")

    vo = VOHibrido(visualize=False, cfg=cfg.get("vo", {}))   # lane-only
    est = Estimator2D()
    ekf = EKF2D(cfg.get("ekf", None))
    ekf.set_state(np.zeros(6))

    lane_ctl = LaneController(cfg["pid"], v_base=cfg["pid"]["v_base"])
    cmd = Commander(cfg["serial"]["port"], cfg["serial"]["baud"])

    # 3) Colas / Hilos
    imu_q = queue.Queue(maxsize=int(cfg["threads"]["imu_hz"] * 4))  # ~4s
    cam_q = queue.Queue(maxsize=1)
    imu_th = IMUThread(imu, imu_q, rate_hz=cfg["threads"]["imu_hz"])
    cam_th = CameraThread(cam, cam_q, target_fps=cfg["threads"]["cam_fps"])
    imu_th.start(); cam_th.start()
    print("[Threads] IMU y Cámara ON")

    ekf_lock = Lock()
    ctrl_period = 1.0 / cfg["threads"]["control_hz"]
    last_ctrl = time.time()
    lane_conf_th = float(cfg["vo"]["lane_conf_threshold"])

    lane = None  # <- inicializar para uso fuera del bloque de frames

    try:
        while True:
            now = time.time()

            # A) Drenar IMU -> propagate -> EKF predict
            while True:
                try:
                    ts, ax, ay, gz = imu_q.get_nowait()
                except queue.Empty:
                    break
                dt = max(1e-3, (now - ts))
                xprop = est.propagate(np.array([ax, ay], dtype=float), gz, dt)
                with ekf_lock:
                    ekf.predict(xprop)

            # B) Si hay frame, correr VO (líneas) -> EKF update heading
            got_frame = False
            try:
                tsf, frame = cam_q.get_nowait()
                got_frame = True
            except queue.Empty:
                pass

            if got_frame:
                res = vo.step(frame)  # VOResult: e_lat, e_head_deg, conf
                lane = res  # para el controlador

                with ekf_lock:
                    if lane and lane.conf > lane_conf_th:
                        th_meas = ekf.x[2] - np.radians(lane.e_head_deg)
                        ekf.update_heading(th_meas)

                # Overlay debug
                with ekf_lock:
                    x, y, th = ekf.x[0], ekf.x[1], ekf.x[2]
                cv2.putText(frame, f"x={x:.2f} y={y:.2f} th={np.degrees(th):.1f}°",
                            (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)
                if lane:
                    cv2.putText(frame, f"e_lat={lane.e_lat:.3f} e_head={lane.e_head_deg:.1f}° conf={lane.conf:.2f}",
                                (8,44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.imshow("WCADS-SLAM MT", frame)
                if cv2.waitKey(1) == 27:
                    break

            # C) Control a tasa fija
            if now - last_ctrl >= ctrl_period:
                last_ctrl = now
                if lane and lane.conf > lane_conf_th:
                    v, omg = lane_ctl.step(lane.e_lat, np.radians(lane.e_head_deg))
                else:
                    v, omg = 0.0, 0.0
                cmd.send_vel(v, omg, dt=ctrl_period)

    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C — salir")

    finally:
        imu_th.stop(); cam_th.stop()
        time.sleep(0.1)
        cam.release()
        cv2.destroyAllWindows()
        print("[OK] Apagado")

if __name__ == "__main__":
    main()
