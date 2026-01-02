from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

from boxing_project.kalman_filter.kalman import KalmanTracker


@dataclass
class Detection:
    """
    Lightweight container for a single detection on a frame.

    Attributes:
      center : (x, y) center of the person (e.g. median of visible keypoints).
      keypoints : (K, 2) array with 2D keypoints; may contain NaNs for low-conf joints.
      kp_conf : (K,) array of keypoint confidences.
      meta : arbitrary metadata (e.g. raw OpenPose JSON, bbox, etc.).
    """
    center: Tuple[float, float]
    keypoints: Optional[np.ndarray] = None
    kp_conf: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Track:
    """
    Single target track managed by the tracker.

    Represents one real-world object (e.g. a person) across multiple frames.
    A track stores:
    - motion state (Kalman filter),
    - reliability over time (hits / confirmation),
    - and object identity via smoothed pose and appearance embeddings (EMA).

    Attributes:
    track_id : int
    Unique ID of the tracked object.

    kf : KalmanTracker
    Motion model used to predict and update the object position.

    min_hits : int
    Number of successful updates required to confirm the track.

    age : int
    Total number of frames since track creation.

    hits : int
    Number of frames where the track was matched with a detection.

    time_since_update : int
    Frames since the last successful update; used for track removal.

    confirmed : bool
    Whether the track is considered reliable (hits >= min_hits).

    last_det_center : Tuple[float, float]
    Last detection center used to update the track.

    last_keypoints : Optional[np.ndarray]
    Last observed keypoints (K, 2), used as a fallback for pose matching.

    last_kp_conf : Optional[np.ndarray]
    Confidence scores for the last keypoints (K,).

    pose_emb_ema : Optional[np.ndarray]
    EMA of pose embeddings, representing the track’s long-term pose memory.

    app_emb_ema : Optional[np.ndarray]
    EMA of appearance embeddings, representing the track’s long-term visual identity.

"""
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

    pose_emb_ema: Optional[np.ndarray] = None
    app_emb_ema: Optional[np.ndarray] = None

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance this track one time step with the Kalman filter.

        Increments:
          - age (total lifetime in frames),
          - time_since_update (frames since last measurement).

        Returns:
          state : np.ndarray (4,) current predicted state [x, y, vx, vy].
          cov   : np.ndarray (4, 4) state covariance matrix.
        """
        self.age += 1
        self.time_since_update += 1
        return self.kf.predict()

    def update(self, det: Detection, ema_alpha: float = 0.9):
        """
        Оновлює трек за matched детекцією.

        Робить:
          1) Kalman update по det.center
          2) оновлює last_* поля (keypoints/conf/center)
          3) EMA-оновлення embeddings (pose/app) з det.meta["e_pose"]/["e_app"]

        Args:
            det: Detection, який Hungarian приписав цьому треку.
            ema_alpha: коеф. EMA (чим ближче до 1, тим повільніше змінюється памʼять треку).
        """
        state, cov = self.kf.update(np.asarray(det.center, dtype=float))
        self.time_since_update = 0
        self.hits += 1
        if not self.confirmed and self.hits >= self.min_hits:
            self.confirmed = True

        self.last_det_center = det.center
        self.last_keypoints = None if det.keypoints is None else np.asarray(det.keypoints, dtype=float)
        self.last_kp_conf = None if det.kp_conf is None else np.asarray(det.kp_conf, dtype=float)

        e_pose = det.meta.get("e_pose", None)
        if e_pose is not None:
            e_pose = np.asarray(e_pose, dtype=np.float32)
            if self.pose_emb_ema is None:
                self.pose_emb_ema = e_pose
            else:
                self.pose_emb_ema = ema_alpha * self.pose_emb_ema + (1.0 - ema_alpha) * e_pose

        e_app = det.meta.get("e_app", None)
        if e_app is not None:
            e_app = np.asarray(e_app, dtype=np.float32)
            if self.app_emb_ema is None:
                self.app_emb_ema = e_app
            else:
                self.app_emb_ema = ema_alpha * self.app_emb_ema + (1.0 - ema_alpha) * e_app

        return state, cov

    def marked_missed(self):
        """
        Placeholder hook for marking a track as 'missed' (unmatched on this frame).

        Currently does nothing, but can be extended in the future if additional
        bookkeeping is needed when a track is not updated by any detection.
        """
        return

    def is_dead(self, max_age: int) -> bool:
        """
        Decide whether this track should be removed.

        A track is considered 'dead' if it has not been updated for more than
        max_age frames.

        Parameters:
          max_age : int
              Maximum allowed time_since_update before removal.

        Returns:
          bool : True if the track should be removed.
        """
        return self.time_since_update > max_age

    @property
    def state(self) -> np.ndarray:
        """
        Current state vector of the Kalman filter.

        Returns:
          np.ndarray (4,) : [x, y, vx, vy].
        """
        return self.kf.get_state()

    def pos(self) -> Tuple[float, float]:
        """
        Convenience method: extract only (x, y) position from the current state.

        Returns:
          (x, y) as floats.
        """
        x, y, *_ = self.state
        return float(x), float(y)

    def project_measurement(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project the current state into measurement space.

        Useful for debugging and gating:
          z_hat = H x
          S     = H P H^T + R

        Returns:
          z_hat : np.ndarray (2, 1)
              Predicted measurement [x, y].
          S : np.ndarray (2, 2)
              Innovation covariance matrix.
        """
        return self.kf.project()
