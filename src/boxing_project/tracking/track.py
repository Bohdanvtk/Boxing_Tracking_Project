from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

from .kalman import KalmanTracker


@dataclass
class Detection:
    center: Tuple[float, float]
    keypoints: Optional[np.ndarray] = None
    kp_conf: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
@dataclass
class Track:
    track_id: int
    kf: KalmanTracker
    min_hits: int

    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    confirmed: bool = False

    last_keypoints: Optional[np.ndarray] = None
    last_kp_conf: Optional[np.ndarray] = None
    last_det_center: Optional[Tuple[float, float]] = None


    def predict(self) -> Tuple[np.ndarray, np.ndarray]:

        self.age += 1
        self.time_since_update += 1
        return self.kf.predict()

    def update(self, det: Detection) -> Tuple[np.ndarray, np.ndarray]:

        state, cov = self.kf.update(np.asarray(det.center, dtype=float))
        self.time_since_update = 0
        self.hits += 1
        if not self.confirmed and self.hits >= self.min_hits:
            self.confirmed = True

        self.last_det_center = det.center
        self.last_keypoints = None if det.keypoints is None else np.asarray(det.keypoints, dtype=float)
        self.last_kp_conf = None if det.kp_conf is None else np.asarray(det.kp_conf, dtype=float)
        return state, cov


    def marked_missed(self):

        return


    def is_dead(self, max_age: int) -> bool:

        return self.time_since_update > max_age


    @property
    def state(self) -> np.ndarray:

        return self.kf.get_state()


    def pos(self) -> Tuple[float, float]:

        x, y, *_ = self.state

        return float(x), float(y)

    def project_measurement(self) -> Tuple[np.ndarray, np.ndarray]:

        return self.kf.project()


