import json, serial, numpy as np

class Commander:
    def __init__(self, port="COM5", baud=115200, timeout=0.05):
        self.ser = serial.Serial(port, baud, timeout=timeout)

    def send_vel(self, v, omega, dt=0.1):
        dx = float(v * dt)
        dth_deg = float(np.degrees(omega * dt))
        cmd = {"cmd":"move", "dx":dx, "dy":0.0, "dtheta_deg":dth_deg}
        self.ser.write((json.dumps(cmd)+"\n").encode())

    def read_odo(self):
        self.ser.write(b'{"cmd":"odo"}\n')
        try:
            line = self.ser.readline().decode().strip()
            if not line: return None
            return json.loads(line).get("odo", None)
        except:
            return None
