import cv2, numpy as np

def open_camera(index=0, w=640, h=480, fps=30):
    cam = cv2.VideoCapture(index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cam.set(cv2.CAP_PROP_FPS, fps)
    return cam

def default_calib(w=640, h=480):
    K = np.array([[520,0,w/2],[0,520,h/2],[0,0,1]],dtype=np.float32)
    dist = np.zeros(5)
    return K, dist
