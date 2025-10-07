# control/config.py
import os
from dotenv import load_dotenv

# Carga variables desde el archivo .env
load_dotenv()

RPI_HOST = os.getenv("RPI_HOST", "127.0.0.1")  # valor por defecto localhost
RPI_PORT = int(os.getenv("RPI_PORT", 65432))
