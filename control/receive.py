 # control/receive.py
from connect import connect
import threading

# Variables globales donde se guardan los últimos datos recibidos
latest_data = {
    "sensor1": None,
    "sensor2": None,
    "video": None,
}

def parse_line(line: str):
    """
    Convierte una línea recibida en datos actualizados.
    Espera mensajes tipo: "sensor1=0.53 sensor2=0.91 video=frame123"
    """
    parts = line.strip().split()
    for p in parts:
        if "=" in p:
            key, val = p.split("=", 1)
            latest_data[key] = val

def receive_lines(sock):
    """
    Escucha mensajes del socket y actualiza 'latest_data'.
    """
    f = sock.makefile("r", encoding="utf-8", newline="\n")
    try:
        for line in f:
            parse_line(line)
            print(f"[RECV] {line.strip()}")
    except Exception as e:
        print(f"[RECV] Error: {e}")
    finally:
        f.close()

def start_receiver_thread():
    """
    Conecta con la Raspi y empieza a escuchar en un hilo separado.
    Retorna el socket (por si querés cerrarlo después).
    """
    sock = connect()
    thread = threading.Thread(target=receive_lines, args=(sock,), daemon=True)
    thread.start()
    print("[RECV] Escuchando datos en segundo plano...")
    return sock

if __name__ == "__main__":
    sock = start_receiver_thread()
    try:
        # Ejemplo de lectura en vivo
        import time
        while True:
            print(f"[DATA] {latest_data}")
            time.sleep(2)
    except KeyboardInterrupt:
        print("[RECV] Detenido por usuario.")
    finally:
        sock.close()
