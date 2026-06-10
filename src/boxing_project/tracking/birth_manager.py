from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from boxing_project.kalman_filter.kalman import KalmanTracker
from .matcher import cosine_similarity, get_common_valid_joints
from .track import Detection, Track


@dataclass
class BirthConfig:
    chi2_gating: float = 9.4877
    max_pending_age: int = 4
    max_pending_misses: int = 2
    very_close_threshold: float = 0.04
    near_threshold: float = 0.15
    pending_motion_threshold: float = 0.20
    normal_confirm_hits: int = 2
    near_confirm_hits: int = 4
    very_close_confirm_hits: int = 999
    easy_birth_track_limit: int = 0
    easy_birth_confirm_hits: int = 1
    emb_ema_alpha: float = 0.9
    min_kp_conf: float = 0.05
    min_core_kps: int = 3
    pose_missing_penalty: float = 0.05
    pose_bad_penalty: float = 0.18
    app_missing_penalty: float = 0.03
    app_bad_penalty: float = 0.12
    app_bad_threshold: float = 0.35
    near_existing_penalty: float = 0.03
    max_birth_score: float = 0.0


@dataclass
class PendingCandidate:
    pending_id: str
    kf: KalmanTracker
    last_detection: Detection
    last_det_idx: int
    last_center: Tuple[float, float]
    last_keypoints: Optional[np.ndarray]
    last_kp_conf: Optional[np.ndarray]
    app_emb_ema: Optional[np.ndarray]
    hits: int
    misses: int
    age: int
    first_seen_frame: int
    last_seen_frame: int
    status: str
    nearest_existing_track_id: Optional[int]
    nearest_existing_d_motion: Optional[float]
    required_confirm_hits: int
    birth_mode: str = "normal"
    very_close_bypass: bool = False


@dataclass
class BirthResult:
    confirmed_birth_det_indices: List[int] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)


class BirthManager:
    def __init__(
        self,
        cfg: BirthConfig,
        dt: float,
        process_var: float,
        measure_var: float,
        p0: float,
        pose_core: Sequence[int],
    ):
        self.cfg = cfg
        self.dt = float(dt)
        self.process_var = float(process_var)
        self.measure_var = float(measure_var)
        self.p0 = float(p0)
        self.pose_core = np.asarray(pose_core, dtype=int)
        self.pending: Dict[str, PendingCandidate] = {}
        self._next_pending_id = 1
        self.easy_birth_created_count = 0

    def reset(self) -> None:
        self.pending.clear()
        self.easy_birth_created_count = 0

    def _mark_confirmed_birth(self, cand: PendingCandidate) -> Tuple[str, bool]:
        birth_mode = "easy_start" if cand.birth_mode == "easy_start" else "normal"
        ignore_overlap = birth_mode == "easy_start"

        if not isinstance(cand.last_detection.meta, dict):
            cand.last_detection.meta = {}

        cand.last_detection.meta["birth_mode"] = birth_mode
        cand.last_detection.meta["ignore_overlap_on_birth"] = bool(ignore_overlap)

        return birth_mode, ignore_overlap

    def _new_pending_id(self) -> str:
        pid = f"P{self._next_pending_id}"
        self._next_pending_id += 1
        return pid

    def _motion_detail(
        self,
        kf: KalmanTracker,
        center: Tuple[float, float],
    ) -> Tuple[float, float]:
        d2 = float(kf.gating_distance(np.asarray(center, dtype=float)))
        return d2, d2 / float(max(self.cfg.chi2_gating, 1e-6))

    def _motion_dist(self, kf: KalmanTracker, center: Tuple[float, float]) -> float:
        return self._motion_detail(kf, center)[1]

    def _nearest_existing(
        self,
        det: Detection,
        tracks: Sequence[Track],
    ) -> Dict[str, Any]:
        best = {
            "nearest_existing_track_idx": None,
            "nearest_existing_track_id": None,
            "nearest_existing_d2": None,
            "nearest_existing_d_motion": float("inf"),
        }

        for idx, t in enumerate(tracks):
            d2, d_motion = self._motion_detail(t.kf, det.center)

            if d_motion < float(best["nearest_existing_d_motion"]):
                best = {
                    "nearest_existing_track_idx": int(idx),
                    "nearest_existing_track_id": int(t.track_id),
                    "nearest_existing_d2": d2,
                    "nearest_existing_d_motion": d_motion,
                }

        return best

    @staticmethod
    def _stable_tracks(tracks: Sequence[Track]) -> List[Track]:
        return [
            t for t in tracks
            if bool(getattr(t, "sub_confirmed", False) or getattr(t, "confirmed", False))
        ]

    def _pending_progress_debug(self, cand: PendingCandidate) -> Dict[str, Any]:
        return {
            "hits": int(cand.hits),
            "misses": int(cand.misses),
            "age": int(cand.age),
            "max_pending_age": int(self.cfg.max_pending_age),
            "max_pending_misses": int(self.cfg.max_pending_misses),
            "age_left": max(0, int(self.cfg.max_pending_age) - int(cand.age)),
            "required_confirm_hits": int(cand.required_confirm_hits),
            "hits_left_to_track": max(
                0,
                int(cand.required_confirm_hits) - int(cand.hits),
            ),
            "ready_for_track": bool(
                int(cand.hits) >= int(cand.required_confirm_hits)
                and int(cand.age) <= int(self.cfg.max_pending_age)
            ),
        }

    def _pending_identity_debug(self, cand: PendingCandidate) -> Dict[str, Any]:
        return {
            "pending_id": cand.pending_id,
            "status": cand.status,
            "birth_mode": cand.birth_mode,
            "very_close_bypass": bool(cand.very_close_bypass),
            "first_seen_frame": int(cand.first_seen_frame),
            "last_seen_frame": int(cand.last_seen_frame),
            "last_det_idx": int(cand.last_det_idx),
            "last_center": tuple(map(float, cand.last_center)),
            "nearest_existing_track_id": cand.nearest_existing_track_id,
            "nearest_existing_d_motion": cand.nearest_existing_d_motion,
        }

    def _pending_frame_attempt_debug(
        self,
        pending_id: str,
        comparisons: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        rows = [
            c for c in comparisons
            if c.get("pending_id") == pending_id
        ]

        scored_rows = [
            c for c in rows
            if c.get("birth_score") is not None
        ]

        best = None
        if scored_rows:
            best = min(scored_rows, key=lambda x: float(x.get("birth_score", float("inf"))))

        return {
            "compared_detection_count": int(len(rows)),
            "motion_passed_count": int(sum(bool(c.get("motion_passed", False)) for c in rows)),
            "score_passed_count": int(
                sum(
                    bool(c.get("score_passed", c.get("motion_passed", False)))
                    for c in rows
                )
            ),
            "best_det_idx": None if best is None else best.get("det_idx"),
            "best_birth_score": None if best is None else best.get("birth_score"),
            "best_reject_reason": None if best is None else best.get("reject_reason"),
            "best_motion_passed": None if best is None else best.get("motion_passed"),
            "best_score_passed": None if best is None else best.get("score_passed"),
        }

    def _closeness_status(self, d_motion: float) -> str:
        if d_motion <= self.cfg.very_close_threshold:
            return "very_close"
        if d_motion <= self.cfg.near_threshold:
            return "near"
        return "far"

    def _pose_penalty(self, cand: PendingCandidate, det: Detection) -> Tuple[str, float]:
        common = get_common_valid_joints(
            cand.last_keypoints,
            det.keypoints,
            cand.last_kp_conf,
            det.kp_conf,
            self.pose_core,
            self.cfg.min_kp_conf,
        )

        if common is None or int(common.size) < int(self.cfg.min_core_kps):
            return "missing", float(self.cfg.pose_missing_penalty)

        v1 = np.asarray(cand.last_keypoints, dtype=float)[common, :2]
        v2 = np.asarray(det.keypoints, dtype=float)[common, :2]

        d = float(np.nanmean(np.linalg.norm(v1 - v2, axis=1)))
        scale = float(np.nanstd(v1) + np.nanstd(v2) + 1e-6)
        score = d / scale

        if score <= 0.8:
            return "ok", 0.0

        return "bad", float(self.cfg.pose_bad_penalty)

    def _app_penalty(self, cand: PendingCandidate, det: Detection) -> Tuple[str, float]:
        emb = det.meta.get("e_app") if isinstance(det.meta, dict) else None

        if cand.app_emb_ema is None or emb is None:
            return "missing", float(self.cfg.app_missing_penalty)

        sim = float(cosine_similarity(np.asarray(cand.app_emb_ema), np.asarray(emb)))
        d_app = (1.0 - sim) / 2.0

        if d_app <= self.cfg.app_bad_threshold:
            return "ok", 0.0

        return "bad", float(self.cfg.app_bad_penalty)

    def update(
        self,
        unmatched_det_indices: Sequence[int],
        detections: Sequence[Detection],
        existing_tracks: Sequence[Track],
        frame_idx: int,
        g: float = 1.0,
    ) -> BirthResult:
        from .birth_update_pipeline import BirthUpdatePipeline

        return BirthUpdatePipeline(self).update(
            unmatched_det_indices=unmatched_det_indices,
            detections=detections,
            existing_tracks=existing_tracks,
            frame_idx=frame_idx,
            g=g,
        )
