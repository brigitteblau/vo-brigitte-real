import math
from .commands import Cmd

def classify_motion(p0, p1, yaw0=None, yaw1=None, v_stop=0.02):
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    v = math.hypot(dx, dy)
    if v < v_stop:
        # opcionalmente, usar yaw para decidir giro en el lugar
        if yaw0 is not None and yaw1 is not None:
            dyaw = math.degrees(yaw1 - yaw0)
            if abs(dyaw) > 10:
                return Cmd.IZQUIERDA if dyaw > 0 else Cmd.DERECHA
        return Cmd.PARA
    ang = math.degrees(math.atan2(dy, dx))
    if -45 <= ang <= 45:
        return Cmd.ADELANTE
    if ang >= 135 or ang <= -135:
        return Cmd.ATRAS
    if 45 < ang < 135:
        return Cmd.IZQUIERDA
    return Cmd.DERECHA
