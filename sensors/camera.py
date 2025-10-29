# sensors/camera.py
import cv2
import numpy as np

def open_camera(index=0, width=640, height=480, fps=30):
    """
    Abre la webcam y devuelve (cam, K, dist).
    K/dist son genéricos; si tenés calibración real, usá la del config.
    """
    cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW ayuda en Windows
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS,          fps)

    K = np.array([[520, 0, width/2],
                  [0, 520, height/2],
                  [0,   0,       1]], dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)
    return cam, K, dist

def default_calib(width=640, height=480):
    """Devuelve (K, dist) genéricos (compatibilidad)."""
    K = np.array([[520, 0, width/2],
                  [0, 520, height/2],
                  [0,   0,       1]], dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)
    return K, dist

