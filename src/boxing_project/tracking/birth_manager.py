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
        stable_tracks = self._stable_tracks(existing_tracks)

        debug: Dict[str, Any] = {
            "frame_idx": int(frame_idx),
            "summary": {
                "incoming_unmatched_count": int(len(unmatched_det_indices)),
                "existing_tracks_count": int(len(existing_tracks)),
                "stable_existing_tracks_count": int(len(stable_tracks)),
                "pending_count_before": int(len(self.pending)),
                "easy_birth_created_count": int(self.easy_birth_created_count),
                "easy_birth_track_limit": int(self.cfg.easy_birth_track_limit),
            },
            "detections": [],
            "confirmed": [],
            "candidate_events": [],
            "pending_comparisons": [],
            "candidates": [],
        }

        for c in self.pending.values():
            c.kf.predict()

        pair_rows = []

        for pid, c in self.pending.items():
            for j in unmatched_det_indices:
                det = detections[int(j)]
                d2, d_motion = self._motion_detail(c.kf, det.center)
                motion_passed = d_motion <= self.cfg.pending_motion_threshold

                comp = {
                    **self._pending_identity_debug(c),
                    **self._pending_progress_debug(c),
                    "det_idx": int(j),
                    "det_center": tuple(map(float, det.center)),
                    "d2": d2,
                    "d_motion": d_motion,
                    "motion_threshold": float(self.cfg.pending_motion_threshold),
                    "motion_passed": bool(motion_passed),
                    "matched_to_pending": False,
                    "pose_status": "not_checked",
                    "pose_penalty": None,
                    "app_status": "not_checked",
                    "app_penalty": None,
                    "near_existing_penalty": None,
                    "birth_score": None,
                    "score_formula": None,
                    "max_birth_score": float(self.cfg.max_birth_score),
                    "score_passed": False,
                    "reject_reason": None,
                }

                if not motion_passed:
                    comp["reject_reason"] = "motion_threshold"
                    debug["pending_comparisons"].append(comp)
                    continue

                pose_status, pose_penalty = self._pose_penalty(c, det)
                app_status, app_penalty = self._app_penalty(c, det)
                near_pen = (
                    float(self.cfg.near_existing_penalty)
                    if c.status == "near_existing"
                    else 0.0
                )

                score = float(d_motion + pose_penalty + app_penalty + near_pen)

                # Optional hard gate after soft pose/app penalties.
                # Motion decides whether the pair can be considered at all;
                # max_birth_score decides whether the combined evidence is still reliable.
                score_threshold = float(self.cfg.max_birth_score)
                score_gate_enabled = score_threshold > 0.0
                score_passed = (not score_gate_enabled) or score <= score_threshold

                comp.update({
                    "pose_status": pose_status,
                    "pose_penalty": pose_penalty,
                    "app_status": app_status,
                    "app_penalty": app_penalty,
                    "near_existing_penalty": near_pen,
                    "birth_score": score,
                    "score_formula": "d_motion + pose_penalty + app_penalty + near_existing_penalty",
                    "max_birth_score": score_threshold,
                    "score_passed": bool(score_passed),
                    "reject_reason": None if score_passed else "birth_score_threshold",
                })

                debug["pending_comparisons"].append(comp)

                if not score_passed:
                    continue

                pair_rows.append(
                    (
                        score,
                        pid,
                        int(j),
                        d2,
                        d_motion,
                        pose_status,
                        pose_penalty,
                        app_status,
                        app_penalty,
                        near_pen,
                    )
                )

        pair_rows.sort(key=lambda x: x[0])

        taken_p, taken_d, matched = set(), set(), {}

        for row in pair_rows:
            _, pid, j, *_ = row

            if pid in taken_p or j in taken_d:
                continue

            taken_p.add(pid)
            taken_d.add(j)
            matched[pid] = row

            for comp in debug["pending_comparisons"]:
                if comp.get("pending_id") == pid and int(comp.get("det_idx", -1)) == int(j):
                    comp["matched_to_pending"] = True

        for pid, c in list(self.pending.items()):
            if pid in matched:
                (
                    score,
                    _,
                    j,
                    d2,
                    d_motion,
                    pose_status,
                    pose_penalty,
                    app_status,
                    app_penalty,
                    near_pen,
                ) = matched[pid]

                det = detections[j]

                c.kf.update(np.asarray(det.center, dtype=float))
                c.last_detection = det
                c.last_det_idx = int(j)
                c.last_center = det.center
                c.last_keypoints = (
                    None
                    if det.keypoints is None
                    else np.asarray(det.keypoints, dtype=float)
                )
                c.last_kp_conf = (
                    None
                    if det.kp_conf is None
                    else np.asarray(det.kp_conf, dtype=float)
                )

                emb = det.meta.get("e_app") if isinstance(det.meta, dict) else None

                if emb is not None:
                    e = np.asarray(emb, dtype=np.float32)
                    c.app_emb_ema = (
                        e
                        if c.app_emb_ema is None
                        else self.cfg.emb_ema_alpha * c.app_emb_ema
                        + (1.0 - self.cfg.emb_ema_alpha) * e
                    )

                c.hits += 1
                c.misses = 0
                c.age += 1
                c.last_seen_frame = int(frame_idx)

                progress = self._pending_progress_debug(c)

                debug["candidate_events"].append({
                    **self._pending_identity_debug(c),
                    "event": "matched_detection",
                    "det_idx": int(j),
                    "matched_det_idx": int(j),
                    "status": c.status,
                    **progress,
                    "ignore_overlap_on_birth": c.birth_mode == "easy_start",
                    "easy_birth_created_count": int(self.easy_birth_created_count),
                    "easy_birth_track_limit": int(self.cfg.easy_birth_track_limit),
                    "matched_birth_score": score,
                    "matched_d_motion": d_motion,
                    "matched_pose_status": pose_status,
                    "matched_pose_penalty": pose_penalty,
                    "matched_app_status": app_status,
                    "matched_app_penalty": app_penalty,
                    "matched_near_existing_penalty": near_pen,
                    "matched_score_passed": True,
                    "matched_reject_reason": None,
                })

                nearest = self._nearest_existing(det, stable_tracks)

                debug["detections"].append({
                    "det_idx": int(j),
                    "det_center": tuple(map(float, det.center)),
                    "action": "pending_matched",
                    "reason": "matched_existing_pending_candidate",
                    **nearest,
                    "closeness_status": self._closeness_status(
                        float(nearest["nearest_existing_d_motion"])
                    ),
                    "very_close_threshold": float(self.cfg.very_close_threshold),
                    "near_threshold": float(self.cfg.near_threshold),
                    "pending_id": pid,
                    "birth_mode": c.birth_mode,
                    "very_close_bypass": bool(c.very_close_bypass),
                    **progress,
                    "easy_birth_created_count": int(self.easy_birth_created_count),
                    "easy_birth_track_limit": int(self.cfg.easy_birth_track_limit),
                    "pending_comparison": {
                        "pending_id": pid,
                        "det_idx": int(j),
                        "d2": d2,
                        "d_motion": d_motion,
                        "motion_threshold": float(self.cfg.pending_motion_threshold),
                        "motion_passed": True,
                        "pose_status": pose_status,
                        "pose_penalty": pose_penalty,
                        "app_status": app_status,
                        "app_penalty": app_penalty,
                        "near_existing_penalty": near_pen,
                        "birth_score": score,
                        "score_formula": "d_motion + pose_penalty + app_penalty + near_existing_penalty",
                        "max_birth_score": float(self.cfg.max_birth_score),
                        "score_passed": True,
                        "reject_reason": None,
                        "matched_to_pending": True,
                    },
                    "d_motion": d_motion,
                    "pose_status": pose_status,
                    "pose_penalty": pose_penalty,
                    "app_status": app_status,
                    "app_penalty": app_penalty,
                    "near_existing_penalty": near_pen,
                    "birth_score": score,
                })

            else:
                c.misses += 1
                c.age += 1

                debug["candidate_events"].append({
                    **self._pending_identity_debug(c),
                    "event": "missed_this_frame",
                    "status": c.status,
                    **self._pending_progress_debug(c),
                    "ignore_overlap_on_birth": c.birth_mode == "easy_start",
                    "easy_birth_created_count": int(self.easy_birth_created_count),
                    "easy_birth_track_limit": int(self.cfg.easy_birth_track_limit),
                    **self._pending_frame_attempt_debug(
                        pending_id=pid,
                        comparisons=debug["pending_comparisons"],
                    ),
                })

            if c.status == "blocked":
                nearest = self._nearest_existing(c.last_detection, stable_tracks)
                nearest_id = nearest["nearest_existing_track_id"]
                nearest_d = float(nearest["nearest_existing_d_motion"])

                c.nearest_existing_track_id = nearest_id
                c.nearest_existing_d_motion = nearest_d

                if nearest_d > self.cfg.very_close_threshold:
                    old_status = c.status
                    c.status = (
                        "near_existing"
                        if nearest_d <= self.cfg.near_threshold
                        else "normal"
                    )

                    if c.status != old_status:
                        debug["candidate_events"].append({
                            **self._pending_identity_debug(c),
                            "event": "status_changed",
                            "from_status": old_status,
                            "to_status": c.status,
                            "nearest_existing_track_id": nearest_id,
                            "nearest_existing_d_motion": nearest_d,
                            **self._pending_progress_debug(c),
                        })

            if c.hits >= c.required_confirm_hits and c.age <= self.cfg.max_pending_age:
                progress = self._pending_progress_debug(c)

                debug["confirmed"].append({
                    **self._pending_identity_debug(c),
                    "source_det_idx": int(c.last_det_idx),
                    "reason": "stable_pending_candidate",
                    "status": c.status,
                    **progress,
                    "hits_left_to_track": 0,
                    "ready_for_track": True,
                    "ignore_overlap_on_birth": c.birth_mode == "easy_start",
                    "easy_birth_created_count": int(self.easy_birth_created_count),
                    "easy_birth_track_limit": int(self.cfg.easy_birth_track_limit),
                    "will_create_new_track": True,
                })

            if c.age > self.cfg.max_pending_age or c.misses > self.cfg.max_pending_misses:
                debug["candidate_events"].append({
                    **self._pending_identity_debug(c),
                    "event": "removed",
                    "reason": "expired_pending_candidate",
                    **self._pending_progress_debug(c),
                })
                self.pending.pop(pid, None)

        confirmed = []
        confirmed_ids = {x["pending_id"] for x in debug["confirmed"]}

        for pid in confirmed_ids:
            c = self.pending.pop(pid, None)

            if c is not None:
                birth_mode, ignore_overlap_on_birth = self._mark_confirmed_birth(c)

                if birth_mode == "easy_start":
                    self.easy_birth_created_count += 1

                for item in debug["confirmed"]:
                    if item.get("pending_id") == pid:
                        item["birth_mode"] = birth_mode
                        item["ignore_overlap_on_birth"] = bool(ignore_overlap_on_birth)

                confirmed.append(int(c.last_det_idx))

        easy_birth_reserved_count = sum(
            1 for c in self.pending.values()
            if c.birth_mode == "easy_start"
        )

        for j in unmatched_det_indices:
            if int(j) in taken_d:
                continue

            det = detections[int(j)]
            nearest = self._nearest_existing(det, stable_tracks)
            nearest_id = nearest["nearest_existing_track_id"]
            nearest_d = float(nearest["nearest_existing_d_motion"])

            easy_birth_available = (
                self.cfg.easy_birth_track_limit > 0
                and self.easy_birth_created_count + easy_birth_reserved_count
                < self.cfg.easy_birth_track_limit
            )

            if easy_birth_available:
                status = "normal"
                action = "pending_created"
                reason = "easy_start"
                req = max(1, int(self.cfg.easy_birth_confirm_hits))
                birth_mode = "easy_start"
                very_close_bypass = nearest_d <= self.cfg.very_close_threshold
                easy_birth_reserved_count += 1

            elif nearest_d <= self.cfg.very_close_threshold:
                status = "blocked"
                action = "blocked_pending"
                reason = "very_close_to_existing_track"
                req = self.cfg.very_close_confirm_hits
                birth_mode = "normal"
                very_close_bypass = False

            elif nearest_d <= self.cfg.near_threshold:
                status = "near_existing"
                action = "pending_created"
                reason = "near_existing_track_requires_more_confirmation"
                req = self.cfg.near_confirm_hits
                birth_mode = "normal"
                very_close_bypass = False

            else:
                status = "normal"
                action = "pending_created"
                reason = "far_from_existing_track"
                req = self.cfg.normal_confirm_hits
                birth_mode = "normal"
                very_close_bypass = False

            pid = self._new_pending_id()

            kf = KalmanTracker(
                x0=[det.center[0], det.center[1], 0.0, 0.0],
                dt=self.dt,
                process_var=self.process_var,
                measure_var=self.measure_var,
                p0=self.p0,
            )

            emb = det.meta.get("e_app") if isinstance(det.meta, dict) else None

            cand = PendingCandidate(
                pid,
                kf,
                det,
                int(j),
                det.center,
                None if det.keypoints is None else np.asarray(det.keypoints, dtype=float),
                None if det.kp_conf is None else np.asarray(det.kp_conf, dtype=float),
                None if emb is None else np.asarray(emb, dtype=np.float32),
                hits=1,
                misses=0,
                age=1,
                first_seen_frame=int(frame_idx),
                last_seen_frame=int(frame_idx),
                status=status,
                nearest_existing_track_id=nearest_id,
                nearest_existing_d_motion=nearest_d,
                required_confirm_hits=int(req),
                birth_mode=birth_mode,
                very_close_bypass=very_close_bypass,
            )

            self.pending[pid] = cand
            progress = self._pending_progress_debug(cand)

            debug["candidate_events"].append({
                **self._pending_identity_debug(cand),
                "event": "created",
                "det_idx": int(j),
                "status": status,
                **progress,
                "ignore_overlap_on_birth": birth_mode == "easy_start",
                "easy_birth_created_count": int(self.easy_birth_created_count),
                "easy_birth_track_limit": int(self.cfg.easy_birth_track_limit),
                "nearest_existing_track_id": nearest_id,
                "nearest_existing_d_motion": nearest_d,
            })

            debug["detections"].append({
                "det_idx": int(j),
                "det_center": tuple(map(float, det.center)),
                "action": action,
                "reason": reason,
                **nearest,
                "closeness_status": self._closeness_status(nearest_d),
                "very_close_threshold": float(self.cfg.very_close_threshold),
                "near_threshold": float(self.cfg.near_threshold),
                "normal_confirm_hits": int(self.cfg.normal_confirm_hits),
                "near_confirm_hits": int(self.cfg.near_confirm_hits),
                "very_close_confirm_hits": int(self.cfg.very_close_confirm_hits),
                "pending_id": pid,
                "birth_mode": birth_mode,
                "very_close_bypass": bool(very_close_bypass),
                **progress,
                "easy_birth_created_count": int(self.easy_birth_created_count),
                "easy_birth_track_limit": int(self.cfg.easy_birth_track_limit),
                "will_create_new_track": False,
            })

            if birth_mode == "easy_start" and cand.hits >= cand.required_confirm_hits:
                self.pending.pop(pid, None)

                birth_mode, ignore_overlap_on_birth = self._mark_confirmed_birth(cand)

                self.easy_birth_created_count += 1
                easy_birth_reserved_count -= 1

                confirmed.append(int(cand.last_det_idx))
                progress = self._pending_progress_debug(cand)

                debug["confirmed"].append({
                    **self._pending_identity_debug(cand),
                    "source_det_idx": int(cand.last_det_idx),
                    "reason": "stable_pending_candidate",
                    "status": cand.status,
                    **progress,
                    "hits_left_to_track": 0,
                    "ready_for_track": True,
                    "birth_mode": birth_mode,
                    "ignore_overlap_on_birth": bool(ignore_overlap_on_birth),
                    "very_close_bypass": bool(cand.very_close_bypass),
                    "easy_birth_created_count": int(self.easy_birth_created_count),
                    "easy_birth_track_limit": int(self.cfg.easy_birth_track_limit),
                    "will_create_new_track": True,
                })

        debug["candidates"] = [
            {
                **self._pending_identity_debug(c),
                **self._pending_progress_debug(c),
                "birth_mode": c.birth_mode,
                "very_close_bypass": bool(c.very_close_bypass),
                "last_det_idx": int(c.last_det_idx),
                "last_center": tuple(map(float, c.last_center)),
                "nearest_existing_track_id": c.nearest_existing_track_id,
                "nearest_existing_d_motion": c.nearest_existing_d_motion,
            }
            for c in self.pending.values()
        ]

        debug["summary"]["confirmed_count"] = int(len(confirmed))
        debug["summary"]["pending_count_after"] = int(len(self.pending))
        debug["summary"]["easy_birth_created_count"] = int(self.easy_birth_created_count)

        return BirthResult(
            confirmed_birth_det_indices=sorted(set(confirmed)),
            debug_info=debug,
        )