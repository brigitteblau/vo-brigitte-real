import numpy as np

class Estimator2D:
    """
    Estado: x = [x, y, theta, vx, vy, omega]
    Integra IMU en el plano (aprox; accel en cuerpo -> mundo vía theta).
    """
    def __init__(self, x0=None):
        self.x = np.zeros(6) if x0 is None else x0.copy()

    def propagate(self, a_xy, gz, dt):
        x,y,th,vx,vy,om = self.x
        th_new = th + gz*dt
        c,s = np.cos(th), np.sin(th)
        ax, ay = a_xy
        # acelerar en mundo
        axw = c*ax - s*ay
        ayw = s*ax + c*ay
        vx_new = vx + axw*dt
        vy_new = vy + ayw*dt
        x_new  = x + vx*dt
        y_new  = y + vy*dt
        self.x = np.array([x_new, y_new, th_new, vx_new, vy_new, gz], dtype=float)
        return self.x
