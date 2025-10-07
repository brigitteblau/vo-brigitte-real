from enum import Enum

class Cmd(str, Enum):
    ADELANTE = "adelante"
    ATRAS = "atrás"
    IZQUIERDA = "izquierda"
    DERECHA = "derecha"
    PARA = "para"
