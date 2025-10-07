from .commands import Cmd

class SimplePlanner:
    """Regla mínima: obstáculo → para + izquierda; libre → adelante."""
    def __init__(self, turn_seconds=0.6, forward_chunk=0.25):
        self.turn_s = turn_seconds
        self.step_s = forward_chunk

    def step(self, obstacle_ahead: bool):
        if obstacle_ahead:
            return [(Cmd.PARA, 0.2), (Cmd.IZQUIERDA, self.turn_s)]
        return [(Cmd.ADELANTE, self.step_s)]
