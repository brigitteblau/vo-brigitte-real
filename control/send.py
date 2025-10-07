from connect import connect

def send_line(sock, text: str):
    """
    Envía una línea de texto (terminada en '\\n') al servidor.
    """
    sock.sendall((text + "\n").encode("utf-8"))
    print(f"[SEND] {text}")

def send_once(text: str):
    sock = connect()
    try:
        send_line(sock, text)
    finally:
        sock.close()
        print("[SEND] Conexión cerrada.")

if __name__ == "__main__":
    # Cambiá por el comando que quieras probar
    send_once("quiero probar si esta bien conectado")
    # Ejemplo secuencia en la misma conexión:
    # sock = connect()
    # try:
    #     send_line(sock, "FWD")
    #     send_line(sock, "LEFT")
    #     send_line(sock, "STOP")
    # finally:
    #     sock.close()
