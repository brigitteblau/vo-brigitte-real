# control/decisor.py
import math
import time

class Decisor:
    """
    Convierte (x,y,yaw) en un comando discreto robusto.
    - Deadband angular y lineal
    - Histeresis: sostiene el último comando por un mínimo de tiempo
    - Rate limit: no spamea al backend
    """
    def __init__(self,
                 lin_deadband=0.2,          # m: ignora movimiento ínfimo
                 yaw_deadband_deg=6.0,       # °: pequeña oscilación no gira
                 min_hold_s=0.5,            # s: mantener comando al menos esto
                 send_hz=5.0,                # Hz: como mucho N cmds/seg
                 x_target=1.0):              # m: ejemplo simple de objetivo x
        self.lin_dead = lin_deadband
        self.yaw_dead = math.radians(yaw_deadband_deg)
        self.min_hold = min_hold_s
        self.min_dt = 1.0 / max(1e-3, send_hz)
        self._last_cmd = None
        self._last_send = 0.0
        self.x_target = x_target

    def _needs_send(self, now, cmd):
        if (now - self._last_send) < self.min_dt:
            return False
        if self._last_cmd is None:
            return True
        if cmd != self._last_cmd and (now - self._last_send) < self.min_hold:
            # no cambies demasiado rápido
            return False
        return True

    def decide(self, x, y, yaw):
        """
        Reglas simples:
        - Si yaw muy a la izquierda  → izquierda
        - Si yaw muy a la derecha    → derecha
        - Si aún no alcanzaste x_target → FWD
        - Si te pasaste mucho         → BACK (o parar si muy cerca)
        """
        # Normalizá yaw a [-pi, pi] por las dudas
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        # Giros primero (prioridad a estabilizar heading)
        if yaw > self.yaw_dead:
            cmd = "izquierda"
        elif yaw < -self.yaw_dead:
            cmd = "derecha"
        else:
            # Heading OK: control lineal en x
            err = self.x_target - x
            if abs(err) < self.lin_dead:
                cmd = "parar"
            elif err > 0:
                cmd = "adelante"
            else:
                cmd = "atras"

        now = time.time()
        if self._needs_send(now, cmd):
            self._last_cmd = cmd
            self._last_send = now
            return cmd, True
        else:
            return cmd, False
