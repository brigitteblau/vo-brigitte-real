# import socket
# from config import RPI_HOST, RPI_PORT

# def connect(host: str = None, port: int = None, timeout: float = 5.0) -> socket.socket:
#     host = host or RPI_HOST
#     port = port or RPI_PORT
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.settimeout(timeout)
#     s.connect((host, port))
#     s.settimeout(None)
#     print(f"[CONNECT] Conectado a {host}:{port}, lam")
#     return s

# if __name__ == "__main__":
#     sock = connect()
#     sock.close()
