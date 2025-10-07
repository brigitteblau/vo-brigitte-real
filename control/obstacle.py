import sys

def detect_dummy_keyboard():
    """Simula obstáculo si apretás Enter (ideal para pruebas)."""
    import select
    rlist, _, _ = select.select([sys.stdin], [], [], 0)
    return bool(rlist)

def detect_from_frames(frame_prev, frame_cur, roi=None, thresh=40_000):
    """Hook para tu lógica real de 'frames comparados'."""
    import cv2
    import numpy as np
    if frame_prev is None or frame_cur is None:
        return False
    if roi is not None:
        x0,y0,x1,y1 = roi
        a = frame_prev[y0:y1, x0:x1]
        b = frame_cur[y0:y1, x0:x1]
    else:
        a, b = frame_prev, frame_cur
    diff = cv2.absdiff(a, b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    score = gray.sum()
    return score > thresh
