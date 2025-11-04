# visualization.py
import numpy as np
import cv2

_WIN = None
_CANVAS = None
_TRAJ = []
_TITLE = "Trayectoria (VO)"
_SCALE = 60.0   # metros->pix aprox
_CENTER = (400, 300)  # centro del canvas (x,y)

def init_traj_view(title="Trayectoria (VO)"):
    global _WIN, _CANVAS, _TITLE, _TRAJ
    _TITLE = title
    _WIN = title
    _CANVAS = np.zeros((600, 800, 3), dtype=np.uint8)  # negro
    _TRAJ = []
    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)

def _world_to_canvas(p):
    # p: (x,y,z) en metros. Dibujamos X-Y (Z se ignora)
    x = int(_CENTER[0] + p[0] * _SCALE)
    y = int(_CENTER[1] - p[1] * _SCALE)
    return (x, y)

def traj_update_from_pose(p_xyz):
    """
    p_xyz: np.array shape (3,) -> (x,y,z)
    """
    global _CANVAS, _TRAJ, _WIN
    if _CANVAS is None:
        return
    _TRAJ.append(np.array(p_xyz, dtype=float))
    img = _CANVAS.copy()

    # Ejes guía
    cv2.line(img, (0, _CENTER[1]), (img.shape[1], _CENTER[1]), (60,60,60), 1)
    cv2.line(img, (_CENTER[0], 0), (_CENTER[0], img.shape[0]), (60,60,60), 1)

    # Dibuja la trayectoria
    for i in range(1, len(_TRAJ)):
        p1 = _world_to_canvas(_TRAJ[i-1])
        p2 = _world_to_canvas(_TRAJ[i])
        cv2.line(img, p1, p2, (0, 255, 255), 2)

    # Punto actual
    if _TRAJ:
        pc = _world_to_canvas(_TRAJ[-1])
        cv2.circle(img, pc, 4, (0, 255, 0), -1)

    cv2.imshow(_WIN, img)
    cv2.waitKey(1)

def save_trajectory_npy(path, traj_list):
    arr = np.vstack(traj_list).astype(np.float32) if len(traj_list) else np.zeros((0,3), np.float32)
    np.save(path, arr)
    return arr
