from dataclasses import dataclass
from typing import Optional

@dataclass
class LaneObs:
    e_lat: float          # error lateral (m aprox o px normalizado)
    e_head_deg: float     # error de heading relativo (grados)
    conf: float           # [0..1]

@dataclass
class ArUcoObs:
    id: int
    tx: float             # m (en frame de cámara o mapeado local)
    ty: float             # m
    tz: float             # m
    yaw: float            # rad

@dataclass
class VOOutput:
    lane: Optional[LaneObs]
    aruco: Optional[ArUcoObs]
