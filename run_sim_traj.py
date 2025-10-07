import argparse, sys, math, numpy as np


# -python run_sim_traj.py trajectory.npy --k 7 --vstop 0.8 --min_dwell 0.4

def normalize_traj(arr):
    """Devuelve [t,x,y,z,roll,pitch,yaw] (N,7) desde formatos comunes."""
    a = np.asarray(arr)
    if a.ndim == 1:
        a = np.stack(a, axis=0)
    n, m = a.shape
    t = np.arange(n, dtype=float)
    x = y = z = roll = pitch = yaw = np.zeros(n, dtype=float)
    if m == 2:              # [x,y]
        x, y = a[:,0], a[:,1]
    elif m == 3:            # [t,x,y]
        t, x, y = a[:,0], a[:,1], a[:,2]
    elif m == 4:            # [t,x,y,yaw]
        t, x, y, yaw = a[:,0], a[:,1], a[:,2], a[:,3]
    elif m >= 7:            # [t,x,y,z,roll,pitch,yaw,...]
        t, x, y, z, roll, pitch, yaw = a[:,0], a[:,1], a[:,2], a[:,3], a[:,4], a[:,5], a[:,6]
    out = np.stack([t,x,y,z,roll,pitch,yaw], axis=1)
    return out

def moving_vec(x, y, i, k):
    """Vector acumulado usando ventana hacia atrás (suaviza)."""
    j = max(0, i - k)
    return (x[i]-x[j], y[i]-y[j])

def ang_deg(dx, dy):
    return math.degrees(math.atan2(dy, dx))

def to_cmd(angle_deg, speed, v_stop, tol=20):
    if speed < v_stop:
        return "para"
    a = angle_deg
    # bins con tolerancia (±tol)
    if -45 - tol <= a <= 45 + tol:          return "adelante"
    if a >= 180 - tol or a <= -180 + tol:   return "atrás"
    if 90 - tol <= a <= 90 + tol:           return "izquierda"
    if -90 - tol <= a <= -90 + tol:         return "derecha"
    # si cae entre bins, elegir el más cercano:
    # mapeo a 0/90/180/-90
    candidates = [0, 90, 180, -90]
    closest = min(candidates, key=lambda c: abs(((a-c+180)%360)-180))
    return to_cmd(closest, max(speed, v_stop+1e-9), v_stop, tol)  # reevalúa ya “forzado”

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help=".npy con trayectoria (N,2|3|4|7)")
    ap.add_argument("--k", type=int, default=5, help="ventana para suavizar (pasos)")
    ap.add_argument("--vstop", type=float, default=0.5, help="umbral de stop (px/s o unidades/s)")
    ap.add_argument("--min_dwell", type=float, default=0.3, help="segundos mínimos para aceptar cambio")
    args = ap.parse_args()

    raw = np.load(args.path, allow_pickle=True)
    traj = normalize_traj(raw)  # (N,7)
    t, x, y, yaw = traj[:,0], traj[:,1], traj[:,2], traj[:,6]

    # Si t no es tiempo real (p.ej. 0..N-1), igual sirve: min_dwell se aplica en steps ~ proporcionales
    last_cmd = None
    last_change_t = t[0] if len(t) else 0.0

    for i in range(1, len(t)):
        dt = max(t[i] - t[i-1], 1e-9)
        dx, dy = moving_vec(x, y, i, args.k)
        # velocidad promedio en la ventana
        path = math.hypot(dx, dy)
        # tiempo de la ventana aprox:
        t_win = max(t[i] - t[max(0, i-args.k)], dt)
        v = path / t_win

        a = ang_deg(dx, dy)
        cand = to_cmd(a, v, v_stop=args.vstop, tol=20)

        # Debounce por dwell
        if cand != last_cmd:
            if (t[i] - last_change_t) >= args.min_dwell or last_cmd is None:
                print(cand)
                last_cmd = cand
                last_change_t = t[i]

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)
