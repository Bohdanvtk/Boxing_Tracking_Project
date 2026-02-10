from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import numpy as np

from boxing_project.kalman_filter.kalman import KalmanTracker


@dataclass
class Detection:
    center: Tuple[float, float]
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    class_id: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Track:
    track_id: int
    kf: KalmanTracker
    min_hits: int

    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    confirmed: bool = False

    last_det_center: Optional[Tuple[float, float]] = None
    last_bbox_xyxy: Optional[tuple[int, int, int, int]] = None
    app_emb_ema: Optional[np.ndarray] = None

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        self.age += 1
        self.time_since_update += 1
        return self.kf.predict()

    def update(self, det: Detection, ema_alpha: float = 0.9, update_app: bool = True):
        state, cov = self.kf.update(np.asarray(det.center, dtype=float))
        self.time_since_update = 0
        self.hits += 1
        if not self.confirmed and self.hits >= self.min_hits:
            self.confirmed = True

        self.last_det_center = det.center
        self.last_bbox_xyxy = det.bbox_xyxy

        if update_app:
            e_app = det.meta.get("e_app", None)
            if e_app is not None:
                e_app = np.asarray(e_app, dtype=np.float32)
                if self.app_emb_ema is None:
                    self.app_emb_ema = e_app
                else:
                    self.app_emb_ema = ema_alpha * self.app_emb_ema + (1.0 - ema_alpha) * e_app

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
