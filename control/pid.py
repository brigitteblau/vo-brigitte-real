import time

class PID:
    def __init__(self, kp, ki, kd, umin=-1e9, umax=1e9):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.umin, self.umax = umin, umax
        self.ei = 0.0
        self.e_prev = 0.0
        self.t0 = time.time()

    def __call__(self, e):
        t = time.time()
        dt = max(1e-3, t - self.t0)
        self.t0 = t
        self.ei += e * dt
        de = (e - self.e_prev) / dt
        self.e_prev = e
        u = self.kp*e + self.ki*self.ei + self.kd*de
        if u < self.umin: u = self.umin
        if u > self.umax: u = self.umax
        return u

class LaneController:
    def __init__(self, cfg, v_base=0.3):
        pL = cfg["lateral"]; pH = cfg["heading"]
        self.pid_lat = PID(pL["kp"], pL["ki"], pL["kd"], pL["umin"], pL["umax"])
        self.pid_head= PID(pH["kp"], pH["ki"], pH["kd"], pH["umin"], pH["umax"])
        self.v_base = v_base

    def step(self, e_lat, e_head_rad):
        omega = self.pid_head(e_head_rad) + self.pid_lat(e_lat)
        v = self.v_base * max(0.2, 1.0 - abs(omega))
        return v, omega
