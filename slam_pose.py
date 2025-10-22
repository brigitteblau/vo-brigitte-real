# ----------------------------------------------------
# Módulo para obtener la pose actual del SLAM.
# Cuando tengas tu VO/SLAM, reemplazá obtener_pose_actual()
# por tu implementación real.
# ----------------------------------------------------
import random

def obtener_pose_actual():
    """
    Devuelve (x, y, yaw). Placeholder para pruebas.
    Reemplazar por lectura real del SLAM.
    """
    x = random.uniform(0, 2)
    y = random.uniform(-1, 1)
    yaw = random.uniform(-3.14, 3.14)
    return x, y, yaw
