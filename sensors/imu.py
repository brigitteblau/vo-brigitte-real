#muy dummy rehacer realmente con el puerto
import time, numpy as np

class IMU:
    def __init__(self, port=None):
        self.bias_gz = 0.0
        self.bias_ax = 0.0
        self.bias_ay = 0.0
        # TODO: abrir puerto/driver real si corresponde

    def calibrate(self, n=400):
        gz, ax, ay = [], [], []
        for _ in range(n):
            g, a = self.read_raw()
            gz.append(g[2]); ax.append(a[0]); ay.append(a[1])
            time.sleep(0.002)
        self.bias_gz = float(np.mean(gz))
        self.bias_ax = float(np.mean(ax))
        self.bias_ay = float(np.mean(ay))

    def read(self):
        (gx, gy, gz), (ax, ay, az) = self.read_raw()
        return np.array([ax-self.bias_ax, ay-self.bias_ay, az]), \
               np.array([gx, gy, gz-self.bias_gz])

    def read_raw(self):
        # TODO: reemplazar por lectura real desde el micro/IMU
        # Ejemplo dummy en reposo
        return (0.0,0.0,0.0), (0.0,0.0,9.81)
