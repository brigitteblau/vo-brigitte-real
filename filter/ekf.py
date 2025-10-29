import numpy as np
import yaml

class EKF2D:
    def __init__(self, cfg=None):
        self.x = np.zeros(6)
        if cfg is None:
            self.P = np.diag([1,1,(5*np.pi/180)**2, 0.1,0.1,(5*np.pi/180)**2])
            self.Q = np.diag([1e-4,1e-4,1e-5, 1e-3,1e-3,1e-4])
            self.R_pose = np.diag([0.02**2,0.02**2,(2*np.pi/180)**2])
            self.R_heading = np.array([[ (5*np.pi/180)**2 ]])
        else:
            self.P = np.diag(cfg["P0_diag"])
            self.Q = np.diag(cfg["Q_diag"])
            self.R_pose = np.diag(cfg["R_pose_diag"])
            self.R_heading = np.array([[cfg["R_heading"]]])

    def set_state(self, x, P=None):
        self.x = x.copy()
        if P is not None: self.P = P.copy()

    def predict(self, x_prop, F=None):
        self.x = x_prop
        self.P = (F if F is not None else np.eye(6)) @ self.P @ (F.T if F is not None else np.eye(6)) + self.Q
        return self.x, self.P

    def update_pose(self, z_xyz):
        # z = [x,y,theta]
        H = np.zeros((3,6))
        H[0,0]=1; H[1,1]=1; H[2,2]=1
        y = z_xyz - self.x[[0,1,2]]
        y[2] = np.arctan2(np.sin(y[2]), np.cos(y[2]))
        S = H @ self.P @ H.T + self.R_pose
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        return self.x, self.P

    def update_heading(self, z_theta):
        H = np.zeros((1,6)); H[0,2]=1
        y = np.array([ np.arctan2(np.sin(z_theta - self.x[2]), np.cos(z_theta - self.x[2])) ])
        S = H @ self.P @ H.T + self.R_heading
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).ravel()
        self.P = (np.eye(6) - K @ H) @ self.P
        return self.x, self.P
