from connect import connect
from send import send_line
from slam_pose import obtener_pose_actual
import time
import math

def decidir_direccion(x, y, yaw):
    """
    Decide un comando de movimiento según la pose actual.
    Editá estas reglas con tu lógica real del SLAM.
    """

    if x < 1.0:
        return "adelante"
    elif yaw > math.radians(90):
        return "izquierda"
    elif yaw < -math.radians(90):
        return "derecha"
    else:
        return "atras"

def main():
    sock = connect()
    try:
        while True:
            x, y, yaw = obtener_pose_actual()
            direccion = decidir_direccion(x, y, yaw)

            # Traducción a comandos que entiende la Raspi:
            if direccion == "adelante":
                send_line(sock, "FWD")
            elif direccion == "atras":
                send_line(sock, "BACK")
            elif direccion == "izquierda":
                send_line(sock, "LEFT")
            elif direccion == "derecha":
                send_line(sock, "RIGHT")
            else:
                send_line(sock, "STOP")

            print(f"[DECISOR] ({x:.2f}, {y:.2f}, yaw={yaw:.2f}) → {direccion}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        send_line(sock, "STOP")
        print("[DECISOR] Interrumpido por el usuario.")
    finally:
        sock.close()
        print("[DECISOR] Conexión cerrada.")

if __name__ == "__main__":
    main()
