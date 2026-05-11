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
    emb_ema_alpha: float = 0.9
    min_kp_conf: float = 0.05
    min_core_kps: int = 3
    pose_missing_penalty: float = 0.05
    pose_bad_penalty: float = 0.18
    app_missing_penalty: float = 0.03
    app_bad_penalty: float = 0.12
    app_bad_threshold: float = 0.35
    near_existing_penalty: float = 0.03


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


@dataclass
class BirthResult:
    confirmed_birth_det_indices: List[int] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)


class BirthManager:
    def __init__(self, cfg: BirthConfig, dt: float, process_var: float, measure_var: float, p0: float, pose_core: Sequence[int]):
        self.cfg = cfg
        self.dt = float(dt)
        self.process_var = float(process_var)
        self.measure_var = float(measure_var)
        self.p0 = float(p0)
        self.pose_core = np.asarray(pose_core, dtype=int)
        self.pending: Dict[str, PendingCandidate] = {}
        self._next_pending_id = 1

    def _new_pending_id(self) -> str:
        pid = f"P{self._next_pending_id}"
        self._next_pending_id += 1
        return pid

    def _motion_detail(self, kf: KalmanTracker, center: Tuple[float, float]) -> Tuple[float, float]:
        d2 = float(kf.gating_distance(np.asarray(center, dtype=float)))
        return d2, d2 / float(max(self.cfg.chi2_gating, 1e-6))

    def _motion_dist(self, kf: KalmanTracker, center: Tuple[float, float]) -> float:
        return self._motion_detail(kf, center)[1]

    def _nearest_existing(self, det: Detection, tracks: Sequence[Track]) -> Dict[str, Any]:
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
        debug: Dict[str, Any] = {
            "frame_idx": int(frame_idx),
            "summary": {
                "incoming_unmatched_count": int(len(unmatched_det_indices)),
                "existing_tracks_count": int(len(existing_tracks)),
                "pending_count_before": int(len(self.pending)),
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
                    "pending_id": pid,
                    "det_idx": int(j),
                    "d2": d2,
                    "d_motion": d_motion,
                    "motion_threshold": float(self.cfg.pending_motion_threshold),
                    "motion_passed": bool(motion_passed),
                    "matched_to_pending": False,
                    "pose_status": "not_checked",
                    "app_status": "not_checked",
                }

                if not motion_passed:
                    debug["pending_comparisons"].append(comp)
                    continue

                pose_status, pose_penalty = self._pose_penalty(c, det)
                app_status, app_penalty = self._app_penalty(c, det)
                near_pen = self.cfg.near_existing_penalty if c.status == "near_existing" else 0.0
                score = float(d_motion + pose_penalty + app_penalty + near_pen)

                comp.update({
                    "pose_status": pose_status,
                    "app_status": app_status,
                    "birth_score": score,
                })
                debug["pending_comparisons"].append(comp)
                pair_rows.append((score, pid, int(j), d2, d_motion, pose_status, pose_penalty, app_status, app_penalty))

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
                _, _, j, d2, d_motion, pose_status, pose_penalty, app_status, app_penalty = matched[pid]
                det = detections[j]

                c.kf.update(np.asarray(det.center, dtype=float))
                c.last_detection = det
                c.last_det_idx = int(j)
                c.last_center = det.center
                c.last_keypoints = None if det.keypoints is None else np.asarray(det.keypoints, dtype=float)
                c.last_kp_conf = None if det.kp_conf is None else np.asarray(det.kp_conf, dtype=float)

                emb = det.meta.get("e_app") if isinstance(det.meta, dict) else None
                if emb is not None:
                    e = np.asarray(emb, dtype=np.float32)
                    c.app_emb_ema = e if c.app_emb_ema is None else (
                        self.cfg.emb_ema_alpha * c.app_emb_ema
                        + (1.0 - self.cfg.emb_ema_alpha) * e
                    )

                c.hits += 1
                c.misses = 0
                c.age += 1
                c.last_seen_frame = int(frame_idx)

                debug["candidate_events"].append({
                    "pending_id": pid,
                    "event": "matched_detection",
                    "det_idx": int(j),
                    "status": c.status,
                    "hits": c.hits,
                    "misses": c.misses,
                    "age": c.age,
                    "required_confirm_hits": c.required_confirm_hits,
                })

                nearest = self._nearest_existing(det, existing_tracks)
                debug["detections"].append({
                    "det_idx": int(j),
                    "det_center": tuple(map(float, det.center)),
                    "action": "pending_matched",
                    "reason": "matched_existing_pending_candidate",
                    **nearest,
                    "closeness_status": self._closeness_status(float(nearest["nearest_existing_d_motion"])),
                    "very_close_threshold": float(self.cfg.very_close_threshold),
                    "near_threshold": float(self.cfg.near_threshold),
                    "pending_id": pid,
                    "hits": c.hits,
                    "misses": c.misses,
                    "age": c.age,
                    "pending_comparison": {
                        "pending_id": pid,
                        "det_idx": int(j),
                        "d2": d2,
                        "d_motion": d_motion,
                        "motion_threshold": float(self.cfg.pending_motion_threshold),
                        "motion_passed": True,
                        "pose_status": pose_status,
                        "app_status": app_status,
                        "birth_score": matched[pid][0],
                        "matched_to_pending": True,
                    },
                    "d_motion": d_motion,
                    "pose_status": pose_status,
                    "pose_penalty": pose_penalty,
                    "app_status": app_status,
                    "app_penalty": app_penalty,
                    "birth_score": matched[pid][0],
                })
            else:
                c.misses += 1
                c.age += 1

                debug["candidate_events"].append({
                    "pending_id": pid,
                    "event": "missed_this_frame",
                    "status": c.status,
                    "hits": c.hits,
                    "misses": c.misses,
                    "age": c.age,
                    "required_confirm_hits": c.required_confirm_hits,
                })

            if c.status == "blocked":
                nearest = self._nearest_existing(c.last_detection, existing_tracks)
                nearest_id = nearest["nearest_existing_track_id"]
                nearest_d = float(nearest["nearest_existing_d_motion"])

                c.nearest_existing_track_id = nearest_id
                c.nearest_existing_d_motion = nearest_d

                if nearest_d > self.cfg.very_close_threshold:
                    old_status = c.status
                    c.status = "near_existing" if nearest_d <= self.cfg.near_threshold else "normal"

                    if c.status != old_status:
                        debug["candidate_events"].append({
                            "pending_id": pid,
                            "event": "status_changed",
                            "from_status": old_status,
                            "to_status": c.status,
                            "nearest_existing_track_id": nearest_id,
                            "nearest_existing_d_motion": nearest_d,
                        })

            if c.hits >= c.required_confirm_hits and c.age <= self.cfg.max_pending_age:
                debug["confirmed"].append({
                    "pending_id": pid,
                    "source_det_idx": int(c.last_det_idx),
                    "reason": "stable_pending_candidate",
                    "status": c.status,
                    "hits": c.hits,
                    "age": c.age,
                    "required_confirm_hits": c.required_confirm_hits,
                    "will_create_new_track": True,
                })

            if c.age > self.cfg.max_pending_age or c.misses > self.cfg.max_pending_misses:
                debug["candidate_events"].append({
                    "pending_id": pid,
                    "event": "removed",
                    "reason": "expired_pending_candidate",
                    "hits": c.hits,
                    "misses": c.misses,
                    "age": c.age,
                })
                self.pending.pop(pid, None)

        confirmed = []
        confirmed_ids = {x["pending_id"] for x in debug["confirmed"]}

        for pid in confirmed_ids:
            c = self.pending.pop(pid, None)
            if c is not None:
                confirmed.append(int(c.last_det_idx))

        for j in unmatched_det_indices:
            if int(j) in taken_d:
                continue

            det = detections[int(j)]
            nearest = self._nearest_existing(det, existing_tracks)
            nearest_id = nearest["nearest_existing_track_id"]
            nearest_d = float(nearest["nearest_existing_d_motion"])

            if nearest_d <= self.cfg.very_close_threshold:
                status = "blocked"
                action = "blocked_pending"
                reason = "very_close_to_existing_track"
                req = self.cfg.very_close_confirm_hits
            elif nearest_d <= self.cfg.near_threshold:
                status = "near_existing"
                action = "pending_created"
                reason = "near_existing_track_requires_more_confirmation"
                req = self.cfg.near_confirm_hits
            else:
                status = "normal"
                action = "pending_created"
                reason = "far_from_existing_track"
                req = self.cfg.normal_confirm_hits

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
            )
            self.pending[pid] = cand

            debug["candidate_events"].append({
                "pending_id": pid,
                "event": "created",
                "det_idx": int(j),
                "status": status,
                "required_confirm_hits": int(req),
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
                "required_confirm_hits": int(req),
                "hits": 1,
                "misses": 0,
                "age": 1,
                "will_create_new_track": False,
            })

        debug["candidates"] = [{
            "pending_id": c.pending_id,
            "status": c.status,
            "hits": c.hits,
            "misses": c.misses,
            "age": c.age,
            "required_confirm_hits": c.required_confirm_hits,
            "last_det_idx": int(c.last_det_idx),
            "last_center": tuple(map(float, c.last_center)),
            "nearest_existing_track_id": c.nearest_existing_track_id,
            "nearest_existing_d_motion": c.nearest_existing_d_motion,
        } for c in self.pending.values()]

        debug["summary"]["confirmed_count"] = int(len(confirmed))
        debug["summary"]["pending_count_after"] = int(len(self.pending))

        return BirthResult(
            confirmed_birth_det_indices=sorted(set(confirmed)),
            debug_info=debug,
        )