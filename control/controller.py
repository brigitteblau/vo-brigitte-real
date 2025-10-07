import time
try:
    import RPi.GPIO as GPIO
except Exception:
    GPIO = None

from .commands import Cmd

class PrintBackend:
    def send(self, cmd: Cmd, dur_s: float = 0.0):
        if dur_s > 0:
            print(f"{cmd} ({dur_s:.2f}s)")
            time.sleep(dur_s)
        else:
            print(cmd)

class GPIOBackend:
    # Ajustá pines BCM a tu driver
    def __init__(self, pin_L1=17, pin_L2=27, pin_R1=22, pin_R2=23):
        if GPIO is None:
            raise RuntimeError("RPi.GPIO no está disponible (corré en la Raspi).")
        self.pL1, self.pL2, self.pR1, self.pR2 = pin_L1, pin_L2, pin_R1, pin_R2
        GPIO.setmode(GPIO.BCM)
        for p in (self.pL1, self.pL2, self.pR1, self.pR2):
            GPIO.setup(p, GPIO.OUT); GPIO.output(p, GPIO.LOW)

    def _drive(self, L_fwd, L_bwd, R_fwd, R_bwd, dur_s):
        GPIO.output(self.pL1, GPIO.HIGH if L_fwd else GPIO.LOW)
        GPIO.output(self.pL2, GPIO.HIGH if L_bwd else GPIO.LOW)
        GPIO.output(self.pR1, GPIO.HIGH if R_fwd else GPIO.LOW)
        GPIO.output(self.pR2, GPIO.HIGH if R_bwd else GPIO.LOW)
        time.sleep(dur_s)
        # frenar
        for p in (self.pL1, self.pL2, self.pR1, self.pR2):
            GPIO.output(p, GPIO.LOW)

    def send(self, cmd: Cmd, dur_s: float = 0.3):
        if cmd == Cmd.PARA:
            self._drive(0,0,0,0, dur_s)
        elif cmd == Cmd.ADELANTE:
            self._drive(1,0, 1,0, dur_s)
        elif cmd == Cmd.ATRAS:
            self._drive(0,1, 0,1, dur_s)
        elif cmd == Cmd.IZQUIERDA:   # giro en el lugar
            self._drive(0,1, 1,0, dur_s)
        elif cmd == Cmd.DERECHA:
            self._drive(1,0, 0,1, dur_s)

    def cleanup(self):
        GPIO.cleanup()
