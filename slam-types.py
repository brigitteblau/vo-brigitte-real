# slam-types.py
import numpy as np

class Keyframe:
    def __init__(self, pose, kpts, map_point_ids):
        self.pose = pose.astype(np.float64)            # (4x4) T_cw
        self.kpts = [np.asarray(k, dtype=np.float64) if k is not None else None for k in kpts]  # [(u,v), ...]
        self.map_point_ids = list(map_point_ids)       # [int or None, ...]

class MapPoint:
    def __init__(self, position, observed_keyframe_ids=None):
        self.position = np.asarray(position, dtype=np.float64)  # (3,)
        self.observed_keyframe_ids = observed_keyframe_ids or []
