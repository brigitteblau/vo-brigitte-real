import threading, time, queue

class IMUThread(threading.Thread):
    def __init__(self, imu, imu_queue: queue.Queue, rate_hz=200):
        super().__init__(daemon=True)
        self.imu = imu
        self.imu_q = imu_queue
        self.rate = rate_hz
        self._stop = threading.Event()

    def run(self):
        period = 1.0 / self.rate
        while not self._stop.is_set():
            ts = time.time()
            a, g = self.imu.read()          # a=[ax,ay,az], g=[gx,gy,gz]
            ax, ay, gz = float(a[0]), float(a[1]), float(g[2])
            try:
                self.imu_q.put_nowait((ts, ax, ay, gz))
            except queue.Full:
                try: self.imu_q.get_nowait()
                except queue.Empty: pass
                self.imu_q.put_nowait((ts, ax, ay, gz))
            time.sleep(period)

    def stop(self): self._stop.set()


class CameraThread(threading.Thread):
    def __init__(self, cam, camera_queue: queue.Queue, target_fps=30):
        super().__init__(daemon=True)
        self.cam = cam
        self.cam_q = camera_queue
        self.fps = target_fps
        self._stop = threading.Event()

    def run(self):
        # Cola size=1: siempre queda el último frame
        while not self._stop.is_set():
            ok, frame = self.cam.read()
            if ok:
                ts = time.time()
                while True:
                    try:
                        self.cam_q.put_nowait((ts, frame))
                        break
                    except queue.Full:
                        try: self.cam_q.get_nowait()
                        except queue.Empty: pass

    def stop(self): self._stop.set()
