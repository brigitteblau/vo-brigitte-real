import time, cv2, queue, yaml, numpy as np
from threading import Lock

from sensors.imu import IMU
from sensors.camera import open_camera, default_calib
from vo.hybrid_vo import HybridVO
from estimator.estimator import Estimator2D
from filter.ekf import EKF2D
from control.pid import LaneController
from control.commander import Commander
from threads import IMUThread, CameraThread

def load_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg()

    # --- Cámara
    cam = open_camera(cfg["camera"]["index"],
                      cfg["camera"]["width"],
                      cfg["camera"]["height"],
                      cfg["camera"]["fps"])
    K = np.array(cfg["camera"]["K"], dtype=np.float32)
    dist = np.array(cfg["camera"]["dist"], dtype=np.float32)
    tag_size = float(cfg["camera"]["aruco_tag_size_m"])

    # --- Sensores y módulos
    imu = IMU()
    print("[IMU] Calibrando...")
    imu.calibrate(n=400)
    print("[IMU] OK")

    vo = HybridVO(K, dist, cfg.get("vo", {}))
    est = Estimator2D()
    ekf = EKF2D(cfg.get("ekf", None))
    ekf.set_state(np.zeros(6))
    lane_ctl = LaneController(cfg["pid"], v_base=cfg["pid"]["v_base"])
    cmd = Commander(cfg["serial"]["port"], cfg["serial"]["baud"])

    # --- Colas/Hilos
    imu_q = queue.Queue(maxsize=int(cfg["threads"]["imu_hz"] * 4)) # ~4s buffer
    cam_q = queue.Queue(maxsize=1)
    imu_th = IMUThread(imu, imu_q, rate_hz=cfg["threads"]["imu_hz"])
    cam_th = CameraThread(cam, cam_q, target_fps=cfg["threads"]["cam_fps"])
    imu_th.start(); cam_th.start()
    print("[Threads] IMU y Cámara ON")

    ekf_lock = Lock()
    ctrl_period = 1.0 / cfg["threads"]["control_hz"]
    last_ctrl = time.time()

    lane_conf_th = float(cfg["vo"]["lane_conf_threshold"])

    try:
        while True:
            now = time.time()

            # 1) Drenar IMU -> Propagate -> Predict
            drained = 0
            while True:
                try:
                    ts, ax, ay, gz = imu_q.get_nowait()
                    drained += 1
                except queue.Empty:
                    break
                dt = max(1e-3, (now - ts))
                xprop = est.propagate(np.array([ax, ay], dtype=float), gz, dt)
                with ekf_lock:
                    ekf.predict(xprop)

            # 2) VO si hay frame
            got_frame = False
            try:
                tsf, frame = cam_q.get_nowait()
                got_frame = True
            except queue.Empty:
                pass

            if got_frame:
                obs = vo.step(frame, tag_size_m=tag_size)
                lane = obs.lane
                ar   = obs.aruco

                with ekf_lock:
                    if ar is not None:
                        z = np.array([ar.tx, ar.ty, ar.yaw], dtype=float)
                        ekf.update_pose(z)
                    if lane and lane.conf > lane_conf_th:
                        th_meas = ekf.x[2] - np.radians(lane.e_head_deg)
                        ekf.update_heading(th_meas)

                # Debug overlay
                with ekf_lock:
                    x,y,th = ekf.x[0], ekf.x[1], ekf.x[2]
                cv2.putText(frame, f"x={x:.2f} y={y:.2f} th={np.degrees(th):.1f}°",
                            (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)
                if lane:
                    cv2.putText(frame, f"e_lat={lane.e_lat:.3f} e_head={lane.e_head_deg:.1f}° conf={lane.conf:.2f}",
                                (8,44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.imshow("WCADS-SLAM MT", frame)
                if cv2.waitKey(1) == 27: break

            # 3) Control
            if now - last_ctrl >= ctrl_period:
                last_ctrl = now
                if got_frame and lane and lane.conf > lane_conf_th:
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
