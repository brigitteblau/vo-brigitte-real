# visualization.py
"""
Módulo de visualización mejorado para Visual Odometry
- Vista 2D tipo mapa con trayectoria
- Vista 3D isométrica con rejilla y ejes
- Panel de información (pose, velocidad, stats)
- Visualización de orientación con flecha direccional
- Historial de trayectoria con desvanecimiento
- Exportación de imágenes y video
"""

import numpy as np
import cv2
import math
from collections import deque
from typing import Optional, Tuple, List

# ============== ESTADO GLOBAL ==============
_WIN_2D = None
_WIN_3D = None
_WIN_INFO = None
_BASE_BG = None        # Fondo estático (mapa) o canvas negro
_CANVAS_2D = None      # Buffer de dibujo 2D por frame
_CANVAS_3D = None      # Buffer de dibujo 3D por frame
_CANVAS_INFO = None    # Panel de información
_TRAJ = []             # Lista de poses (x,y,z)
_TRAJ_HISTORY = deque(maxlen=500)  # Historial con fade
_TITLE = "Trayectoria (VO)"
_POSE_HISTORY = deque(maxlen=10)   # Para calcular velocidad
_FRAME_COUNT = 0
_START_TIME = None

# Transformación mundo->imagen (2D)
_MPP = 0.02            # meters per pixel
_ORIGIN_PX = (400, 300)
_ROT_RAD = 0.0
_COSR, _SINR = 1.0, 0.0

# Configuración 3D
_SCALE_3D = 50.0       # píxeles por metro en vista 3D
_CENTER_3D = (400, 300)
_PITCH_3D = 35.0       # ángulo de inclinación (grados)
_YAW_3D = 45.0         # ángulo de rotación (grados)

# Colores
COLOR_TRAJ = (0, 255, 255)      # Cyan
COLOR_CURRENT = (0, 255, 0)     # Verde
COLOR_GRID = (80, 80, 80)       # Gris oscuro
COLOR_AXES_X = (0, 0, 255)      # Rojo
COLOR_AXES_Y = (0, 255, 0)      # Verde
COLOR_AXES_Z = (255, 0, 0)      # Azul
COLOR_TEXT = (255, 255, 255)    # Blanco
COLOR_BG = (20, 20, 20)         # Gris muy oscuro


# ============== TRANSFORMACIONES ==============

def _set_transform(meters_per_pixel=0.02, origin_px=(400, 300), rot_deg=0.0):
    """Configura la transformación mundo->imagen para vista 2D"""
    global _MPP, _ORIGIN_PX, _ROT_RAD, _COSR, _SINR
    _MPP = float(meters_per_pixel)
    _ORIGIN_PX = (int(origin_px[0]), int(origin_px[1]))
    _ROT_RAD = math.radians(rot_deg)
    _COSR = math.cos(_ROT_RAD)
    _SINR = math.sin(_ROT_RAD)


def _world_to_img_2d(p_xy: Tuple[float, float]) -> Tuple[int, int]:
    """Transforma coordenadas mundo (x,y) a píxeles imagen 2D"""
    x, y = float(p_xy[0]), float(p_xy[1])
    xr = _COSR * x - _SINR * y
    yr = _SINR * x + _COSR * y
    u = int(_ORIGIN_PX[0] + xr / _MPP)
    v = int(_ORIGIN_PX[1] - yr / _MPP)
    return (u, v)


def _world_to_img_3d(p_xyz: Tuple[float, float, float]) -> Tuple[int, int]:
    """
    Transforma coordenadas mundo (x,y,z) a píxeles con proyección isométrica
    Aplica rotación yaw y pitch para vista 3D
    """
    x, y, z = float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2])
    
    # Rotación yaw (alrededor de Z)
    yaw_rad = math.radians(_YAW_3D)
    x_rot = x * math.cos(yaw_rad) - y * math.sin(yaw_rad)
    y_rot = x * math.sin(yaw_rad) + y * math.cos(yaw_rad)
    
    # Rotación pitch (alrededor de X)
    pitch_rad = math.radians(_PITCH_3D)
    y_final = y_rot * math.cos(pitch_rad) - z * math.sin(pitch_rad)
    z_final = y_rot * math.sin(pitch_rad) + z * math.cos(pitch_rad)
    
    # Proyección a pantalla
    u = int(_CENTER_3D[0] + x_rot * _SCALE_3D)
    v = int(_CENTER_3D[1] - z_final * _SCALE_3D)  # Z hacia arriba en pantalla
    
    return (u, v)


# ============== INICIALIZACIÓN ==============

def init_traj_view(title="Trayectoria (VO)", show_3d=True, show_info=True):
    """
    Inicializa las ventanas de visualización
    
    Args:
        title: Título de las ventanas
        show_3d: Si mostrar vista 3D
        show_info: Si mostrar panel de información
    """
    global _WIN_2D, _WIN_3D, _WIN_INFO, _BASE_BG, _CANVAS_2D, _CANVAS_3D
    global _CANVAS_INFO, _TITLE, _TRAJ, _TRAJ_HISTORY, _START_TIME
    
    _TITLE = title
    _WIN_2D = title + " - Vista 2D"
    _TRAJ = []
    _TRAJ_HISTORY.clear()
    _POSE_HISTORY.clear()
    _START_TIME = None
    
    # Vista 2D (mapa)
    if _BASE_BG is None:
        _BASE_BG = np.zeros((600, 800, 3), dtype=np.uint8)
        _BASE_BG[:] = COLOR_BG
    _CANVAS_2D = _BASE_BG.copy()
    cv2.namedWindow(_WIN_2D, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WIN_2D, 800, 600)
    
    # Vista 3D
    if show_3d:
        _WIN_3D = title + " - Vista 3D"
        _CANVAS_3D = np.zeros((600, 800, 3), dtype=np.uint8)
        _CANVAS_3D[:] = COLOR_BG
        cv2.namedWindow(_WIN_3D, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_WIN_3D, 800, 600)
    
    # Panel de información
    if show_info:
        _WIN_INFO = title + " - Info"
        _CANVAS_INFO = np.zeros((300, 400, 3), dtype=np.uint8)
        _CANVAS_INFO[:] = COLOR_BG
        cv2.namedWindow(_WIN_INFO, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_WIN_INFO, 400, 300)


def set_map(image_path: str,
            meters_per_pixel: float = 0.02,
            origin_px: tuple = (400, 300),
            rot_deg: float = 0.0,
            fit_height: Optional[int] = None,
            fit_width: Optional[int] = None):
    """
    Carga imagen de mapa como fondo para vista 2D
    
    Args:
        image_path: Ruta a la imagen del mapa
        meters_per_pixel: Escala del mapa (metros por píxel)
        origin_px: Píxel correspondiente al origen (0,0) del mundo
        rot_deg: Rotación para alinear ejes (grados, CCW)
        fit_height: Altura objetivo para redimensionar
        fit_width: Ancho objetivo para redimensionar
    """
    global _BASE_BG, _CANVAS_2D
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] No se pudo abrir el mapa: {image_path}, usando canvas negro")
        return
    
    if fit_height is not None:
        scale = fit_height / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * scale), fit_height), 
                        interpolation=cv2.INTER_AREA)
    if fit_width is not None:
        scale = fit_width / img.shape[1]
        img = cv2.resize(img, (fit_width, int(img.shape[0] * scale)), 
                        interpolation=cv2.INTER_AREA)
    
    _set_transform(meters_per_pixel, origin_px, rot_deg)
    _BASE_BG = img
    _CANVAS_2D = _BASE_BG.copy()


# ============== DIBUJADO 2D ==============

def _draw_grid_2d(canvas: np.ndarray, grid_size: float = 1.0, max_range: float = 20.0):
    """Dibuja rejilla en vista 2D"""
    h, w = canvas.shape[:2]
    
    # Líneas verticales (X)
    for x in np.arange(-max_range, max_range + grid_size, grid_size):
        p1 = _world_to_img_2d((x, -max_range))
        p2 = _world_to_img_2d((x, max_range))
        if 0 <= p1[0] < w and 0 <= p2[0] < w:
            cv2.line(canvas, p1, p2, COLOR_GRID, 1, cv2.LINE_AA)
    
    # Líneas horizontales (Y)
    for y in np.arange(-max_range, max_range + grid_size, grid_size):
        p1 = _world_to_img_2d((-max_range, y))
        p2 = _world_to_img_2d((max_range, y))
        if 0 <= p1[1] < h and 0 <= p2[1] < h:
            cv2.line(canvas, p1, p2, COLOR_GRID, 1, cv2.LINE_AA)


def _draw_axes_2d(canvas: np.ndarray, length: float = 2.0):
    """Dibuja ejes de coordenadas en vista 2D"""
    origin = _world_to_img_2d((0.0, 0.0))
    x_end = _world_to_img_2d((length, 0.0))
    y_end = _world_to_img_2d((0.0, length))
    
    # Eje X (rojo)
    cv2.arrowedLine(canvas, origin, x_end, COLOR_AXES_X, 2, 
                    tipLength=0.15, line_type=cv2.LINE_AA)
    cv2.putText(canvas, "X", (x_end[0] + 10, x_end[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_AXES_X, 2)
    
    # Eje Y (verde)
    cv2.arrowedLine(canvas, origin, y_end, COLOR_AXES_Y, 2, 
                    tipLength=0.15, line_type=cv2.LINE_AA)
    cv2.putText(canvas, "Y", (y_end[0], y_end[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_AXES_Y, 2)


def _draw_trajectory_2d(canvas: np.ndarray, traj_list: List[np.ndarray], 
                        fade: bool = True):
    """Dibuja la trayectoria en vista 2D con efecto de desvanecimiento opcional"""
    if len(traj_list) < 2:
        return
    
    n = len(traj_list)
    for i in range(1, n):
        p1 = _world_to_img_2d(traj_list[i - 1][:2])
        p2 = _world_to_img_2d(traj_list[i][:2])
        
        if fade:
            # Desvanecimiento: más reciente = más brillante
            alpha = i / n
            color = tuple(int(c * alpha) for c in COLOR_TRAJ)
            thickness = int(1 + alpha * 2)
        else:
            color = COLOR_TRAJ
            thickness = 2
        
        cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)


def _draw_current_pose_2d(canvas: np.ndarray, pose: np.ndarray, 
                          yaw: Optional[float] = None):
    """Dibuja la pose actual con flecha direccional"""
    pos_2d = _world_to_img_2d(pose[:2])
    
    # Círculo para posición
    cv2.circle(canvas, pos_2d, 6, COLOR_CURRENT, -1, cv2.LINE_AA)
    cv2.circle(canvas, pos_2d, 8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Flecha direccional si se proporciona yaw
    if yaw is not None:
        arrow_length = 40  # píxeles
        end_x = int(pos_2d[0] + arrow_length * math.cos(yaw))
        end_y = int(pos_2d[1] - arrow_length * math.sin(yaw))
        cv2.arrowedLine(canvas, pos_2d, (end_x, end_y), (255, 255, 0), 3, 
                       tipLength=0.3, line_type=cv2.LINE_AA)


# ============== DIBUJADO 3D ==============

def _draw_grid_3d(canvas: np.ndarray, grid_size: float = 1.0, 
                  max_range: float = 10.0, levels: int = 5):
    """Dibuja rejilla 3D con múltiples niveles"""
    # Rejilla en plano XY para diferentes alturas Z
    for z_level in np.linspace(-2, 2, levels):
        alpha = 1.0 - abs(z_level) / 3.0  # Desvanecimiento con altura
        color = tuple(int(c * alpha) for c in COLOR_GRID)
        
        # Líneas paralelas a X
        for y in np.arange(-max_range, max_range + grid_size, grid_size):
            p1 = _world_to_img_3d((-max_range, y, z_level))
            p2 = _world_to_img_3d((max_range, y, z_level))
            cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        
        # Líneas paralelas a Y
        for x in np.arange(-max_range, max_range + grid_size, grid_size):
            p1 = _world_to_img_3d((x, -max_range, z_level))
            p2 = _world_to_img_3d((x, max_range, z_level))
            cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)


def _draw_axes_3d(canvas: np.ndarray, length: float = 3.0):
    """Dibuja ejes 3D de coordenadas"""
    origin = _world_to_img_3d((0.0, 0.0, 0.0))
    x_end = _world_to_img_3d((length, 0.0, 0.0))
    y_end = _world_to_img_3d((0.0, length, 0.0))
    z_end = _world_to_img_3d((0.0, 0.0, length))
    
    # Eje X (rojo)
    cv2.arrowedLine(canvas, origin, x_end, COLOR_AXES_X, 3, 
                    tipLength=0.15, line_type=cv2.LINE_AA)
    cv2.putText(canvas, "X", (x_end[0] + 15, x_end[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_AXES_X, 2)
    
    # Eje Y (verde)
    cv2.arrowedLine(canvas, origin, y_end, COLOR_AXES_Y, 3, 
                    tipLength=0.15, line_type=cv2.LINE_AA)
    cv2.putText(canvas, "Y", (y_end[0] + 15, y_end[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_AXES_Y, 2)
    
    # Eje Z (azul)
    cv2.arrowedLine(canvas, origin, z_end, COLOR_AXES_Z, 3, 
                    tipLength=0.15, line_type=cv2.LINE_AA)
    cv2.putText(canvas, "Z", (z_end[0] + 15, z_end[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_AXES_Z, 2)


def _draw_trajectory_3d(canvas: np.ndarray, traj_list: List[np.ndarray], 
                        fade: bool = True):
    """Dibuja la trayectoria en vista 3D isométrica"""
    if len(traj_list) < 2:
        return
    
    n = len(traj_list)
    for i in range(1, n):
        p1 = _world_to_img_3d(traj_list[i - 1])
        p2 = _world_to_img_3d(traj_list[i])
        
        if fade:
            alpha = i / n
            color = tuple(int(c * alpha) for c in COLOR_TRAJ)
            thickness = int(1 + alpha * 3)
        else:
            color = COLOR_TRAJ
            thickness = 2
        
        cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
    
    # Puntos en la trayectoria cada N poses
    for i in range(0, n, max(1, n // 20)):
        pt = _world_to_img_3d(traj_list[i])
        cv2.circle(canvas, pt, 3, (100, 200, 255), -1, cv2.LINE_AA)


def _draw_current_pose_3d(canvas: np.ndarray, pose: np.ndarray):
    """Dibuja la pose actual en vista 3D"""
    pos_3d = _world_to_img_3d(pose)
    
    # Esfera para posición actual
    cv2.circle(canvas, pos_3d, 8, COLOR_CURRENT, -1, cv2.LINE_AA)
    cv2.circle(canvas, pos_3d, 10, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Línea vertical desde suelo
    ground_pos = _world_to_img_3d((pose[0], pose[1], 0.0))
    cv2.line(canvas, ground_pos, pos_3d, (150, 150, 150), 1, 
             cv2.LINE_AA)
    cv2.circle(canvas, ground_pos, 4, (100, 100, 100), -1, cv2.LINE_AA)


# ============== PANEL DE INFORMACIÓN ==============

def _calculate_velocity(pose_history: deque) -> Tuple[float, float, float]:
    """Calcula velocidad aproximada desde historial de poses"""
    if len(pose_history) < 2:
        return 0.0, 0.0, 0.0
    
    dt = 0.1  # Asumimos ~10 Hz
    recent = list(pose_history)
    
    vx = (recent[-1][0] - recent[0][0]) / (dt * len(recent))
    vy = (recent[-1][1] - recent[0][1]) / (dt * len(recent))
    vz = (recent[-1][2] - recent[0][2]) / (dt * len(recent))
    
    return vx, vy, vz


def _draw_info_panel(canvas: np.ndarray, pose: np.ndarray, 
                     frame_count: int, velocity: Tuple[float, float, float],
                     total_dist: float):
    canvas[:] = COLOR_BG
    y_offset = 30
    line_height = 25

    def draw_text(text: str, y, color=COLOR_TEXT):
        y_int = int(round(y))
        cv2.putText(
            canvas,
            text,
            (10, y_int),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    # Título
    cv2.putText(canvas, "ESTADO DEL SISTEMA", (10, y_offset - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # Posición
    draw_text(f"Posicion (m):", y_offset + line_height * 1, (100, 200, 255))
    draw_text(f"  X: {pose[0]:7.3f}", y_offset + line_height * 2)
    draw_text(f"  Y: {pose[1]:7.3f}", y_offset + line_height * 3)
    draw_text(f"  Z: {pose[2]:7.3f}", y_offset + line_height * 4)

    # Velocidad
    vx, vy, vz = velocity
    v_total = math.sqrt(vx**2 + vy**2 + vz**2)
    draw_text(f"Velocidad (m/s):", y_offset + line_height * 5.5, (100, 200, 255))
    draw_text(f"  Vx: {vx:6.3f}", y_offset + line_height * 6.5)
    draw_text(f"  Vy: {vy:6.3f}", y_offset + line_height * 7.5)
    draw_text(f"  |V|: {v_total:6.3f}", y_offset + line_height * 8.5, (0, 255, 0))

    # Estadísticas
    draw_text(f"Frames: {frame_count}", y_offset + line_height * 10)
    draw_text(f"Distancia: {total_dist:.2f} m", y_offset + line_height * 11)


# ============== ACTUALIZACIÓN PRINCIPAL ==============

def traj_update_from_pose(p_xyz: np.ndarray, yaw: Optional[float] = None):
    """
    Actualiza todas las visualizaciones con la nueva pose
    
    Args:
        p_xyz: Pose (x, y, z) en metros
        yaw: Ángulo yaw en radianes (opcional)
    """
    global _CANVAS_2D, _CANVAS_3D, _CANVAS_INFO, _TRAJ, _TRAJ_HISTORY
    global _POSE_HISTORY, _FRAME_COUNT, _START_TIME
    
    if _BASE_BG is None:
        return
    
    # Inicializar tiempo si es el primer frame
    if _START_TIME is None:
        import time
        _START_TIME = time.time()
    
    _FRAME_COUNT += 1
    pose = np.array(p_xyz, dtype=float)
    _TRAJ.append(pose)
    _TRAJ_HISTORY.append(pose)
    _POSE_HISTORY.append(pose)
    
    # Calcular distancia total recorrida
    total_dist = 0.0
    if len(_TRAJ) > 1:
        for i in range(1, len(_TRAJ)):
            total_dist += np.linalg.norm(_TRAJ[i] - _TRAJ[i-1])
    
    # ========== VISTA 2D ==========
    _CANVAS_2D = _BASE_BG.copy()
    _draw_grid_2d(_CANVAS_2D, grid_size=1.0, max_range=20.0)
    _draw_axes_2d(_CANVAS_2D, length=2.0)
    _draw_trajectory_2d(_CANVAS_2D, list(_TRAJ_HISTORY), fade=True)
    _draw_current_pose_2d(_CANVAS_2D, pose, yaw)
    
    # Texto overlay
    cv2.putText(_CANVAS_2D, f"Frame: {_FRAME_COUNT}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
    cv2.putText(_CANVAS_2D, f"Pos: ({pose[0]:.2f}, {pose[1]:.2f})", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
    
    cv2.imshow(_WIN_2D, _CANVAS_2D)
    
    # ========== VISTA 3D ==========
    if _WIN_3D is not None:
        _CANVAS_3D = np.zeros((600, 800, 3), dtype=np.uint8)
        _CANVAS_3D[:] = COLOR_BG
        
        _draw_grid_3d(_CANVAS_3D, grid_size=2.0, max_range=10.0, levels=5)
        _draw_axes_3d(_CANVAS_3D, length=3.0)
        _draw_trajectory_3d(_CANVAS_3D, list(_TRAJ_HISTORY), fade=True)
        _draw_current_pose_3d(_CANVAS_3D, pose)
        
        # Controles
        info_text = [
            "Controles 3D:",
            "Q/A: Pitch +/-",
            "W/S: Yaw +/-",
            "E/D: Escala +/-"
        ]
        for i, txt in enumerate(info_text):
            cv2.putText(_CANVAS_3D, txt, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        cv2.imshow(_WIN_3D, _CANVAS_3D)
    
    # ========== PANEL INFO ==========
    if _WIN_INFO is not None:
        velocity = _calculate_velocity(_POSE_HISTORY)
        _draw_info_panel(_CANVAS_INFO, pose, _FRAME_COUNT, velocity, total_dist)
        cv2.imshow(_WIN_INFO, _CANVAS_INFO)
    
    # Procesar teclas para controles 3D
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        _adjust_3d_view(pitch_delta=5)
    elif key == ord('a'):
        _adjust_3d_view(pitch_delta=-5)
    elif key == ord('w'):
        _adjust_3d_view(yaw_delta=5)
    elif key == ord('s'):
        _adjust_3d_view(yaw_delta=-5)
    elif key == ord('e'):
        _adjust_3d_view(scale_factor=1.1)
    elif key == ord('d'):
        _adjust_3d_view(scale_factor=0.9)


def _adjust_3d_view(pitch_delta=0, yaw_delta=0, scale_factor=1.0):
    """Ajusta los parámetros de la vista 3D"""
    global _PITCH_3D, _YAW_3D, _SCALE_3D
    
    _PITCH_3D = max(-80, min(80, _PITCH_3D + pitch_delta))
    _YAW_3D = (_YAW_3D + yaw_delta) % 360
    _SCALE_3D = max(10, min(200, _SCALE_3D * scale_factor))


# ============== EXPORTACIÓN ==============

def save_trajectory_npy(path: str, traj_list: Optional[List] = None) -> np.ndarray:
    """Guarda la trayectoria en formato NumPy"""
    if traj_list is None:
        traj_list = _TRAJ
    
    arr = np.vstack(traj_list).astype(np.float32) if len(traj_list) else np.zeros((0, 3), np.float32)
    np.save(path, arr)
    print(f"[INFO] Trayectoria guardada: {path} ({len(arr)} poses)")
    return arr


def save_visualization_image(path_2d: Optional[str] = None, 
                             path_3d: Optional[str] = None,
                             path_info: Optional[str] = None):
    """Guarda capturas de las visualizaciones actuales"""
    if path_2d and _CANVAS_2D is not None:
        cv2.imwrite(path_2d, _CANVAS_2D)
        print(f"[INFO] Vista 2D guardada: {path_2d}")
    
    if path_3d and _CANVAS_3D is not None:
        cv2.imwrite(path_3d, _CANVAS_3D)
        print(f"[INFO] Vista 3D guardada: {path_3d}")
    
    if path_info and _CANVAS_INFO is not None:
        cv2.imwrite(path_info, _CANVAS_INFO)
        print(f"[INFO] Panel info guardado: {path_info}")


def cleanup():
    """Cierra todas las ventanas de visualización"""
    cv2.destroyAllWindows()
    print("[INFO] Visualización cerrada")


# ============== UTILIDADES ADICIONALES ==============

def set_3d_view_params(scale: float = 50.0, pitch: float = 35.0, 
                       yaw: float = 45.0, center: Tuple[int, int] = (400, 300)):
    """
    Configura parámetros de la vista 3D
    
    Args:
        scale: Píxeles por metro
        pitch: Ángulo de inclinación en grados
        yaw: Ángulo de rotación en grados
        center: Centro de la proyección en píxeles
    """
    global _SCALE_3D, _PITCH_3D, _YAW_3D, _CENTER_3D
    _SCALE_3D = scale
    _PITCH_3D = pitch
    _YAW_3D = yaw
    _CENTER_3D = center


def get_trajectory() -> np.ndarray:
    """Retorna la trayectoria actual como array NumPy"""
    if len(_TRAJ) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack(_TRAJ).astype(np.float32)


def clear_trajectory():
    """Limpia la trayectoria almacenada"""
    global _TRAJ, _TRAJ_HISTORY, _POSE_HISTORY, _FRAME_COUNT
    _TRAJ.clear()
    _TRAJ_HISTORY.clear()
    _POSE_HISTORY.clear()
    _FRAME_COUNT = 0
    print("[INFO] Trayectoria limpiada")


# ============== MODO COMPARACIÓN ==============

def draw_comparison(traj_estimated: List[np.ndarray], 
                   traj_ground_truth: List[np.ndarray],
                   canvas: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Dibuja comparación entre trayectoria estimada y ground truth
    
    Args:
        traj_estimated: Lista de poses estimadas
        traj_ground_truth: Lista de poses ground truth
        canvas: Canvas existente o None para crear uno nuevo
        
    Returns:
        Canvas con la comparación dibujada
    """
    if canvas is None:
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        canvas[:] = COLOR_BG
    
    # Dibujar rejilla y ejes
    _draw_grid_2d(canvas, grid_size=1.0, max_range=20.0)
    _draw_axes_2d(canvas, length=2.0)
    
    # Dibujar ground truth en verde
    if len(traj_ground_truth) > 1:
        for i in range(1, len(traj_ground_truth)):
            p1 = _world_to_img_2d(traj_ground_truth[i-1][:2])
            p2 = _world_to_img_2d(traj_ground_truth[i][:2])
            cv2.line(canvas, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Dibujar estimada en cyan
    if len(traj_estimated) > 1:
        for i in range(1, len(traj_estimated)):
            p1 = _world_to_img_2d(traj_estimated[i-1][:2])
            p2 = _world_to_img_2d(traj_estimated[i][:2])
            cv2.line(canvas, p1, p2, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Leyenda
    cv2.rectangle(canvas, (10, 10), (200, 80), (0, 0, 0), -1)
    cv2.rectangle(canvas, (10, 10), (200, 80), (255, 255, 255), 1)
    cv2.putText(canvas, "Comparacion", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.line(canvas, (20, 45), (50, 45), (0, 255, 0), 2)
    cv2.putText(canvas, "Ground Truth", (60, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.line(canvas, (20, 65), (50, 65), (255, 255, 0), 2)
    cv2.putText(canvas, "Estimada", (60, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    return canvas


def calculate_trajectory_error(traj_estimated: List[np.ndarray], 
                               traj_ground_truth: List[np.ndarray]) -> dict:
    """
    Calcula métricas de error entre trayectoria estimada y ground truth
    
    Returns:
        Diccionario con métricas: ATE, RPE, etc.
    """
    if len(traj_estimated) != len(traj_ground_truth):
        print("[WARN] Las trayectorias tienen diferentes longitudes")
        n = min(len(traj_estimated), len(traj_ground_truth))
        traj_estimated = traj_estimated[:n]
        traj_ground_truth = traj_ground_truth[:n]
    
    est = np.array(traj_estimated)
    gt = np.array(traj_ground_truth)
    
    # Absolute Trajectory Error (ATE)
    ate = np.linalg.norm(est - gt, axis=1)
    
    # Relative Pose Error (RPE)
    rpe = []
    for i in range(1, len(est)):
        delta_est = est[i] - est[i-1]
        delta_gt = gt[i] - gt[i-1]
        rpe.append(np.linalg.norm(delta_est - delta_gt))
    
    metrics = {
        'ate_mean': float(np.mean(ate)),
        'ate_std': float(np.std(ate)),
        'ate_rmse': float(np.sqrt(np.mean(ate**2))),
        'ate_max': float(np.max(ate)),
        'rpe_mean': float(np.mean(rpe)) if rpe else 0.0,
        'rpe_std': float(np.std(rpe)) if rpe else 0.0,
        'final_error': float(np.linalg.norm(est[-1] - gt[-1])),
        'num_poses': len(est)
    }
    
    return metrics


def print_trajectory_stats():
    """Imprime estadísticas de la trayectoria actual"""
    if len(_TRAJ) < 2:
        print("[INFO] No hay suficientes poses para estadísticas")
        return
    
    traj = np.array(_TRAJ)
    
    # Distancia total
    total_dist = 0.0
    for i in range(1, len(traj)):
        total_dist += np.linalg.norm(traj[i] - traj[i-1])
    
    # Rango de movimiento
    x_range = traj[:, 0].max() - traj[:, 0].min()
    y_range = traj[:, 1].max() - traj[:, 1].min()
    z_range = traj[:, 2].max() - traj[:, 2].min()
    
    # Velocidad promedio
    velocities = []
    for i in range(1, len(traj)):
        dt = 0.1  # Asumiendo ~10 Hz
        v = np.linalg.norm(traj[i] - traj[i-1]) / dt
        velocities.append(v)
    
    print("\n" + "="*50)
    print("ESTADÍSTICAS DE TRAYECTORIA")
    print("="*50)
    print(f"Número de poses:     {len(traj)}")
    print(f"Distancia total:     {total_dist:.2f} m")
    print(f"Rango X:             {x_range:.2f} m")
    print(f"Rango Y:             {y_range:.2f} m")
    print(f"Rango Z:             {z_range:.2f} m")
    print(f"Posición inicial:    ({traj[0][0]:.2f}, {traj[0][1]:.2f}, {traj[0][2]:.2f})")
    print(f"Posición final:      ({traj[-1][0]:.2f}, {traj[-1][1]:.2f}, {traj[-1][2]:.2f})")
    if velocities:
        print(f"Velocidad promedio:  {np.mean(velocities):.2f} m/s")
        print(f"Velocidad máxima:    {np.max(velocities):.2f} m/s")
    print("="*50 + "\n")


# ============== EXPORTACIÓN A VIDEO ==============

class TrajectoryVideoWriter:
    """Clase para grabar las visualizaciones a video"""
    
    def __init__(self, output_path: str, fps: int = 30, 
                 record_2d: bool = True, record_3d: bool = True):
        """
        Inicializa el grabador de video
        
        Args:
            output_path: Ruta base para los videos (sin extensión)
            fps: Frames por segundo
            record_2d: Grabar vista 2D
            record_3d: Grabar vista 3D
        """
        self.output_path = output_path
        self.fps = fps
        self.record_2d = record_2d
        self.record_3d = record_3d
        
        self.writer_2d = None
        self.writer_3d = None
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        if record_2d and _CANVAS_2D is not None:
            h, w = _CANVAS_2D.shape[:2]
            self.writer_2d = cv2.VideoWriter(
                f"{output_path}_2d.mp4", fourcc, fps, (w, h)
            )
            print(f"[INFO] Grabando vista 2D: {output_path}_2d.mp4")
        
        if record_3d and _CANVAS_3D is not None:
            h, w = _CANVAS_3D.shape[:2]
            self.writer_3d = cv2.VideoWriter(
                f"{output_path}_3d.mp4", fourcc, fps, (w, h)
            )
            print(f"[INFO] Grabando vista 3D: {output_path}_3d.mp4")
    
    def write_frame(self):
        """Escribe el frame actual a los videos"""
        if self.writer_2d is not None and _CANVAS_2D is not None:
            self.writer_2d.write(_CANVAS_2D)
        
        if self.writer_3d is not None and _CANVAS_3D is not None:
            self.writer_3d.write(_CANVAS_3D)
    
    def release(self):
        """Finaliza la grabación y cierra los archivos"""
        if self.writer_2d is not None:
            self.writer_2d.release()
            print("[INFO] Video 2D guardado")
        
        if self.writer_3d is not None:
            self.writer_3d.release()
            print("[INFO] Video 3D guardado")


# ============== MINI MAPA ==============

def create_minimap(traj_list: List[np.ndarray], 
                   size: Tuple[int, int] = (200, 200)) -> np.ndarray:
    """
    Crea un minimapa de la trayectoria completa
    
    Args:
        traj_list: Lista de poses
        size: Tamaño del minimapa (width, height)
        
    Returns:
        Imagen del minimapa
    """
    minimap = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    minimap[:] = (30, 30, 30)
    
    if len(traj_list) < 2:
        return minimap
    
    traj = np.array(traj_list)
    
    # Normalizar coordenadas al tamaño del minimapa
    x_min, x_max = traj[:, 0].min(), traj[:, 0].max()
    y_min, y_max = traj[:, 1].min(), traj[:, 1].max()
    
    margin = 10
    if x_max - x_min > 0 and y_max - y_min > 0:
        scale_x = (size[0] - 2 * margin) / (x_max - x_min)
        scale_y = (size[1] - 2 * margin) / (y_max - y_min)
        scale = min(scale_x, scale_y)
        
        # Centrar
        offset_x = (size[0] - (x_max - x_min) * scale) / 2
        offset_y = (size[1] - (y_max - y_min) * scale) / 2
        
        # Dibujar trayectoria
        for i in range(1, len(traj)):
            x1 = int((traj[i-1][0] - x_min) * scale + offset_x)
            y1 = int((traj[i-1][1] - y_min) * scale + offset_y)
            x2 = int((traj[i][0] - x_min) * scale + offset_x)
            y2 = int((traj[i][1] - y_min) * scale + offset_y)
            
            cv2.line(minimap, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
        
        # Posición actual
        x_curr = int((traj[-1][0] - x_min) * scale + offset_x)
        y_curr = int((traj[-1][1] - y_min) * scale + offset_y)
        cv2.circle(minimap, (x_curr, y_curr), 3, (0, 255, 0), -1)
    
    # Borde
    cv2.rectangle(minimap, (0, 0), (size[0]-1, size[1]-1), (100, 100, 100), 1)
    
    return minimap