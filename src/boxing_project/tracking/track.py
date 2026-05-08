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

      meta : arbitrary metadata (e.g. raw OpenPose JSON, bbox, errors, etc.).
    """
    center: Tuple[float, float]
    keypoints: Optional[np.ndarray] = None
    kp_conf: Optional[np.ndarray] = None


    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Track:
    """
    Single target track managed by MultiObjectTracker.

    A Track represents one local object identity inside one tracking epoch.

    It stores only local state for one target:
      - Kalman / motion state,
      - pose state from the latest matched detection,
      - reliability counters,
      - appearance identity memory,
      - local overlap/cooldown bookkeeping.

    Important architecture rule:
      Track does NOT decide global overlap-group behavior.
      Track does NOT decide which other tracks disappeared or returned.
      MultiObjectTracker owns that global logic.

    Track only stores:
      - with whom it overlapped on the previous frame,
      - whether its appearance memory is currently frozen,
      - and which source track caused that freeze.

    Motion/pose update rule:
      A matched detection always updates:
        - Kalman state,
        - time_since_update,
        - hits / confirmed,
        - last_det_center,
        - last_keypoints / last_kp_conf.

    Appearance update rule:
      Appearance memory is updated only when:
        - update_app is True,
        - det.meta["e_app"] exists,
        - current detection has no dangerous adaptive overlap,
        - this track has no active freeze source.

    bbox_quality is intentionally NOT used to block appearance updates.
    If e_app exists and the crop is not blocked by overlap/cooldown,
    the appearance EMA can be updated.

    Attributes:
        track_id:
            Persistent local track id inside the current epoch.

        epoch_id:
            Tracking epoch id. Increased after reset/shot-boundary reset.

        kf:
            KalmanTracker used for motion prediction/update.

        min_hits:
            Number of matched detections required before confirmed=True.

        age:
            Total lifetime in frames since this track was created.

        hits:
            Number of frames where this track was matched to a detection.

        time_since_update:
            Number of frames since the last matched detection.
            Incremented by predict(), reset to 0 by update().

        confirmed:
            True after hits >= min_hits.

        last_det_center:
            Last matched detection center.

        last_keypoints:
            Last matched detection keypoints, used later by pose matching.

        last_kp_conf:
            Last matched detection keypoint confidences.

        app_emb_ema:
            Smoothed appearance embedding for identity matching.
            Protected from contaminated overlap crops.

        app_emb_history:
            Accepted appearance embeddings used later for global clustering.

        left_glove_features_history / right_glove_features_history / shorts_features_history:
            Accepted local color/part features used for later identity grouping.

        overlap_group_ids:
            Track ids that overlapped with this track on the previous frame.
            MultiObjectTracker writes this field after converting detection overlap
            relations into track ids.

        freeze_sources:
            Source-based appearance cooldown map:
                source_track_id -> frames_left

            Example:
                freeze_sources = {3: 4}

            Meaning:
                this track is frozen for 4 more frames because Track 3 disappeared
                after being in an overlap group with this track.

        freeze_frames_left:
            Convenience summary of freeze_sources.
            Usually max(freeze_sources.values()) or 0 if no source is active.
    """
    track_id: int
    kf: KalmanTracker
    min_hits: int
    epoch_id: int = 1

    age: int = 0

    hits: int = 0
    time_since_update: int = 0
    confirmed: bool = False

    last_keypoints: Optional[np.ndarray] = None
    last_kp_conf: Optional[np.ndarray] = None
    last_det_center: Optional[Tuple[float, float]] = None

    app_emb_ema: Optional[np.ndarray] = None
    app_emb_history: list[np.ndarray] = field(default_factory=list)

    left_glove_features_history: list[np.ndarray] = field(default_factory=list)
    right_glove_features_history: list[np.ndarray] = field(default_factory=list)
    shorts_features_history: list[np.ndarray] = field(default_factory=list)

    # Track ids that overlapped with this track on the previous frame.
    # MultiObjectTracker writes this field; Track only stores it.
    overlap_group_ids: set[int] = field(default_factory=set)

    # Source-based appearance freeze.
    # Key: source track_id that disappeared after overlap.
    # Value: frames left.
    freeze_sources: Dict[int, int] = field(default_factory=dict)
    freeze_frames_left: int = 0

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

    def update(
            self,
            det: Detection,
            ema_alpha: float = 0.9,
            update_motion: bool = True,
            update_pose: bool = True,
            update_app: bool = True,
            adaptive_overlap_center_near: float = 0.55,
            adaptive_overlap_center_mid: float = 0.85,
            adaptive_overlap_center_far: float = 1.20,
            adaptive_overlap_iou_near: float = 0.03,
            adaptive_overlap_iou_mid: float = 0.06,
            adaptive_overlap_iou_far: float = 0.08,
            adaptive_overlap_iou_default: float = 0.12,
    ):
        # Adaptive overlap can skip contaminated motion/pose/app updates.
        max_overlap_iou = float(det.meta.get("max_overlap_iou", 0.0))
        center_dist_norm = float(det.meta.get("min_center_dist_norm", float("inf")))

        if center_dist_norm <= adaptive_overlap_center_near:
            active_overlap_threshold = adaptive_overlap_iou_near
            adaptive_overlap_zone = "near"

        elif center_dist_norm <= adaptive_overlap_center_mid:
            active_overlap_threshold = adaptive_overlap_iou_mid
            adaptive_overlap_zone = "mid"

        elif center_dist_norm <= adaptive_overlap_center_far:
            active_overlap_threshold = adaptive_overlap_iou_far
            adaptive_overlap_zone = "far"

        else:
            active_overlap_threshold = adaptive_overlap_iou_default
            adaptive_overlap_zone = "default"
        # Disabled adaptive overlap means no overlap-based update blocking.
        current_overlap = max_overlap_iou > float(active_overlap_threshold)

        det.meta["adaptive_overlap_zone"] = adaptive_overlap_zone
        det.meta["active_overlap_threshold"] = float(active_overlap_threshold)
        det.meta["min_center_dist_norm"] = center_dist_norm
        det.meta["center_dist_norm_det_idx"] = det.meta.get("center_dist_norm_det_idx")

        if current_overlap:
            update_motion = False
            update_pose = False
            update_app = False
        elif not bool(update_motion):
            update_pose = False
            update_app = False

        disabled_reasons = list(det.meta.get("track_update_disabled_reasons", []) or [])
        if current_overlap and "overlap_update_disabled" not in disabled_reasons:
            disabled_reasons.append("overlap_update_disabled")

        det.meta["track_update_motion_allowed"] = bool(update_motion)
        det.meta["track_update_pose_allowed"] = bool(update_pose)
        det.meta["track_update_app_requested"] = bool(update_app)
        det.meta["track_update_disabled_reasons"] = disabled_reasons
        det.meta["track_update_skip_reason"] = ",".join(disabled_reasons) if disabled_reasons else None
        det.meta["track_update_skipped"] = not (bool(update_motion) and bool(update_pose) and bool(update_app))
        det.meta["track_update_fully_skipped"] = not bool(update_motion)
        det.meta["track_update_partially_skipped"] = bool(update_motion) and (not bool(update_pose) or not bool(update_app))
        det.meta["track_match_had_overlap"] = bool(current_overlap)

        if update_motion:
            state, cov = self.kf.update(np.asarray(det.center, dtype=float))
            self.time_since_update = 0
            self.hits += 1
            if not self.confirmed and self.hits >= self.min_hits:
                self.confirmed = True
            self.last_det_center = det.center
        else:
            state, cov = self.kf.get_state(), self.kf.get_cov()

        if update_pose and not current_overlap:
            self.last_keypoints = None if det.keypoints is None else np.asarray(det.keypoints, dtype=float)
            self.last_kp_conf = None if det.kp_conf is None else np.asarray(det.kp_conf, dtype=float)

        self._sync_freeze_counter()

        e_app = det.meta.get("e_app", None)
        freeze_active = self.is_frozen()

        allow_app_update = (
            bool(update_app)
            and e_app is not None
            and not current_overlap
            and not freeze_active
        )

        det.meta["track_app_update_allowed"] = bool(allow_app_update)
        det.meta["track_freeze_frames_left"] = int(self.freeze_frames_left)
        det.meta["track_freeze_source_ids"] = sorted(int(source_id) for source_id in self.freeze_sources.keys())

        if allow_app_update:
            e_app = np.asarray(e_app, dtype=np.float32)

            if self.app_emb_ema is None:
                self.app_emb_ema = e_app
            else:
                self.app_emb_ema = ema_alpha * self.app_emb_ema + (1.0 - ema_alpha) * e_app

            self.app_emb_history.append(e_app.copy())

            lf = det.meta.get("left_glove_features")
            if lf is not None:
                self.left_glove_features_history.append(lf)

            rf = det.meta.get("right_glove_features")
            if rf is not None:
                self.right_glove_features_history.append(rf)

            sf = det.meta.get("shorts_features")
            if sf is not None:
                self.shorts_features_history.append(sf)

            det.meta["track_app_update_block_reason"] = None
        else:
            if current_overlap:
                reason = "current_detection_overlap"
            elif not bool(update_app):
                reason = "update_app_false"
            elif e_app is None:
                reason = "missing_e_app"
            elif freeze_active:
                reason = "freeze_source_active"
            else:
                reason = "unknown"

            det.meta["track_app_update_block_reason"] = reason

        return state, cov

    @staticmethod
    def overlap_group(det: Detection) -> list[int]:
        # Return det_idx values whose relation is risky under adaptive overlap.
        group: list[int] = []

        for rel in det.meta.get("overlap_relations", []) or []:
            try:
                if bool(rel.get("adaptive_overlap_risk", False)):
                    group.append(int(rel.get("det_idx")))
            except Exception:
                continue

        return group

    def _sync_freeze_counter(self):
        if self.freeze_sources:
            self.freeze_frames_left = max(int(v) for v in self.freeze_sources.values())
        else:
            self.freeze_frames_left = 0

    def is_frozen(self) -> bool:
        self._sync_freeze_counter()
        return self.freeze_frames_left > 0

    def set_cooldown(self, source_track_id: int, frames: int):
        # Local setter. The tracker decides when to call it.
        source_track_id = int(source_track_id)
        frames = max(0, int(frames))
        if frames <= 0:
            return

        old_value = int(self.freeze_sources.get(source_track_id, 0))
        self.freeze_sources[source_track_id] = max(old_value, frames)
        self._sync_freeze_counter()

    def clear_freeze_source(self, source_track_id: int):
        # Clear only one source. Example: if source M returned,
        # clear freeze_sources[M] but keep other sources.
        self.freeze_sources.pop(int(source_track_id), None)
        self._sync_freeze_counter()

    def decrease_freeze(self, exclude_sources: Optional[set[int]] = None):
        # Decrease all source cooldowns by one frame.
        # exclude_sources are new sources started on this same frame.
        exclude_sources = {int(x) for x in (exclude_sources or set())}

        for source_track_id, frames_left in list(self.freeze_sources.items()):
            source_track_id = int(source_track_id)

            if source_track_id in exclude_sources:
                continue

            frames_left = int(frames_left) - 1

            if frames_left <= 0:
                self.freeze_sources.pop(source_track_id, None)
            else:
                self.freeze_sources[source_track_id] = frames_left

        self._sync_freeze_counter()

    def marked_missed(self):
        # Compatibility hook. The group-level missed/freeze logic lives in MultiObjectTracker.
        return

    def is_dead(self, max_age: int, max_confirmed_age) -> bool:
        """
        Determine whether this track should be removed.

        A track is normally considered dead if it has not received
        an update for more than `max_age` frames.

        Parameters:
            max_age : int
                Maximum allowed number of frames without update.

        Returns:
            bool
                True if the track should be removed.
        """

        if self.confirmed:
            return  self.time_since_update > max_confirmed_age

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


    def return_available_kps(self, keypoints):
        pass
