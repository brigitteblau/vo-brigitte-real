# visualization.py
import numpy as np
import cv2
import math

_WIN = None
_BASE_BG = None        # fondo estático (mapa) o canvas negro
_CANVAS = None         # buffer de dibujo por frame
_TRAJ = []
_TITLE = "Trayectoria (VO)"

# --- Transformación mundo->imagen ---
# Mundo en metros (x,y). Imagen en píxeles (u,v).
_MPP = 0.02            # meters per pixel (default: 1px = 0.02 m -> 50 px por metro)
_ORIGIN_PX = (400, 300)  # píxel en imagen que corresponde a (x,y)=(0,0) mundo
_ROT_RAD = 0.0         # rotación (mundo->imagen) en rad (positiva CCW)
_COSR, _SINR = 1.0, 0.0

def _set_transform(meters_per_pixel=0.02, origin_px=(400,300), rot_deg=0.0):
    global _MPP, _ORIGIN_PX, _ROT_RAD, _COSR, _SINR
    _MPP = float(meters_per_pixel)
    _ORIGIN_PX = (int(origin_px[0]), int(origin_px[1]))
    _ROT_RAD = math.radians(rot_deg)
    _COSR = math.cos(_ROT_RAD)
    _SINR = math.sin(_ROT_RAD)

def init_traj_view(title="Trayectoria (VO)"):
    global _WIN, _BASE_BG, _CANVAS, _TITLE, _TRAJ
    _TITLE = title
    _WIN = title
    # si no hay mapa aún, armamos un lienzo negro default
    _BASE_BG = np.zeros((600, 800, 3), dtype=np.uint8)
    _CANVAS = _BASE_BG.copy()
    _TRAJ = []
    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)

def set_map(image_path: str,
            meters_per_pixel: float = 0.02,
            origin_px: tuple = (400, 300),
            rot_deg: float = 0.0,
            fit_height: int | None = None,
            fit_width: int | None = None):
    """
    Carga imagen de mapa como fondo y define la transformación mundo->imagen.
    - meters_per_pixel: cuántos metros "reales" hay por píxel del mapa
    - origin_px: píxel (u0,v0) donde cae el punto (x,y)=(0,0) de tu VO
    - rot_deg: rotación para alinear ejes VO con el mapa (CCW)
    - fit_height/fit_width: si querés redimensionar el mapa de entrada
    """
    global _BASE_BG, _CANVAS
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] No pude abrir el mapa: {image_path}, uso canvas negro")
        return
    if fit_height is not None:
        scale = fit_height / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*scale), fit_height), interpolation=cv2.INTER_AREA)
    if fit_width is not None:
        scale = fit_width / img.shape[1]
        img = cv2.resize(img, (fit_width, int(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)

    _set_transform(meters_per_pixel, origin_px, rot_deg)
    _BASE_BG = img
    _CANVAS = _BASE_BG.copy()

def _world_to_img_xy(p_xy):
    """
    p_xy: (x,y) en metros (plano). Devuelve (u,v) en píxeles (int).
    Aplica rotación (mundo->imagen), escala (m->px) y offset de origen.
    """
    x, y = float(p_xy[0]), float(p_xy[1])
    # rotación (CCW): [u';v'] = R * [x;y]
    xr =  _COSR * x - _SINR * y
    yr =  _SINR * x + _COSR * y
    u = int(_ORIGIN_PX[0] + xr / _MPP)
    v = int(_ORIGIN_PX[1] - yr / _MPP)  # eje v hacia abajo en imagen
    return (u, v)

def traj_update_from_pose(p_xyz):
    """
    p_xyz: array-like (3,) -> (x,y,z) en metros
    Dibuja la trayectoria sobre el mapa (si existe) o canvas negro.
    """
    global _CANVAS, _TRAJ, _WIN, _BASE_BG
    if _BASE_BG is None:
        return
    _TRAJ.append(np.array(p_xyz, dtype=float))
    _CANVAS = _BASE_BG.copy()

    # Rejilla/opcional: dibujar ejes del origen definido
    # Ejes guía locales (10 m izquierda/derecha y 10 m arriba/abajo)
    try:
        u0, v0 = _world_to_img_xy((0.0, 0.0))
        uX, vX = _world_to_img_xy((1.0, 0.0))
        uY, vY = _world_to_img_xy((0.0, 1.0))
        cv2.arrowedLine(_CANVAS, (u0, v0), (uX, vX if False else vX), (0,0,255), 1, tipLength=0.03)  # rojo X
        cv2.arrowedLine(_CANVAS, (u0, v0), (uY, vY), (0,255,0), 1, tipLength=0.03)                 # verde Y
    except Exception:
        pass

    # Trazo de la trayectoria
    for i in range(1, len(_TRAJ)):
        p1 = _world_to_img_xy(_TRAJ[i-1][:2])
        p2 = _world_to_img_xy(_TRAJ[i][:2])
        cv2.line(_CANVAS, p1, p2, (0, 255, 255), 2)

    # Punto actual
    if _TRAJ:
        pc = _world_to_img_xy(_TRAJ[-1][:2])
        cv2.circle(_CANVAS, pc, 4, (0, 255, 0), -1)

    cv2.imshow(_WIN, _CANVAS)
    cv2.waitKey(1)

def save_trajectory_npy(path, traj_list):
    arr = np.vstack(traj_list).astype(np.float32) if len(traj_list) else np.zeros((0,3), np.float32)
    np.save(path, arr)
    return arr
