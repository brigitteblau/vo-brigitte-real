# control/config.py
# import os
# from dotenv import load_dotenv
# load_dotenv()

# RPI_HOST = os.getenv("RPI_HOST", "127.0.0.1")
# RPI_PORT = int(os.getenv("RPI_PORT", 65432))
# config.py — solo cámara local en la Raspberry
CAM_INDEX = 0          # normalmente 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30
OUTPUT_DIR = "captures"  # carpeta para guardar fotos
