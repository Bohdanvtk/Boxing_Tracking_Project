from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from boxing_project.kalman_filter.kalman import KalmanTracker
from .track import Detection, Track
from .birth_manager import BirthResult, PendingCandidate


class BirthUpdatePipeline:
    def __init__(self, manager):
        object.__setattr__(self, "_manager", manager)

    def __getattr__(self, name):
        return getattr(self._manager, name)

    def __setattr__(self, name, value):
        if name == "_manager":
            object.__setattr__(self, name, value)
        else:
            setattr(self._manager, name, value)

    def _build_update_debug(
        self,
        unmatched_det_indices: Sequence[int],
        existing_tracks: Sequence[Track],
        stable_tracks: Sequence[Track],
        frame_idx: int,
    ) -> Dict[str, Any]:
        return {
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

    def _predict_pending(self) -> None:
        for cand in self.pending.values():
            cand.kf.predict()

    def _score_pending_pair(
        self,
        cand: PendingCandidate,
        det: Detection,
        det_idx: int,
    ) -> Tuple[Optional[Tuple[Any, ...]], Dict[str, Any]]:
        d2, d_motion = self._motion_detail(cand.kf, det.center)
        motion_passed = d_motion <= self.cfg.pending_motion_threshold

        comp = {
            **self._pending_identity_debug(cand),
            **self._pending_progress_debug(cand),
            "det_idx": int(det_idx),
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
            return None, comp

        pose_status, pose_penalty = self._pose_penalty(cand, det)
        app_status, app_penalty = self._app_penalty(cand, det)
        near_pen = (
            float(self.cfg.near_existing_penalty)
            if cand.status == "near_existing"
            else 0.0
        )
        score = float(d_motion + pose_penalty + app_penalty + near_pen)

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

        if not score_passed:
            return None, comp

        row = (
            score,
            cand.pending_id,
            int(det_idx),
            d2,
            d_motion,
            pose_status,
            pose_penalty,
            app_status,
            app_penalty,
            near_pen,
        )
        return row, comp

    def _build_pending_pair_rows(
        self,
        unmatched_det_indices: Sequence[int],
        detections: Sequence[Detection],
        debug: Dict[str, Any],
    ) -> List[Tuple[Any, ...]]:
        pair_rows: List[Tuple[Any, ...]] = []

        for _, cand in self.pending.items():
            for j in unmatched_det_indices:
                row, comp = self._score_pending_pair(cand, detections[int(j)], int(j))
                debug["pending_comparisons"].append(comp)
                if row is not None:
                    pair_rows.append(row)

        pair_rows.sort(key=lambda x: x[0])
        return pair_rows

    def _select_pending_matches(
        self,
        pair_rows: Sequence[Tuple[Any, ...]],
        debug: Dict[str, Any],
    ) -> Tuple[set, Dict[str, Tuple[Any, ...]]]:
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

        return taken_d, matched

    def _apply_pending_match(
        self,
        cand: PendingCandidate,
        det: Detection,
        det_idx: int,
        frame_idx: int,
    ) -> None:
        cand.kf.update(np.asarray(det.center, dtype=float))
        cand.last_detection = det
        cand.last_det_idx = int(det_idx)
        cand.last_center = det.center
        cand.last_keypoints = None if det.keypoints is None else np.asarray(det.keypoints, dtype=float)
        cand.last_kp_conf = None if det.kp_conf is None else np.asarray(det.kp_conf, dtype=float)

        emb = det.meta.get("e_app") if isinstance(det.meta, dict) else None
        if emb is not None:
            e = np.asarray(emb, dtype=np.float32)
            cand.app_emb_ema = (
                e
                if cand.app_emb_ema is None
                else self.cfg.emb_ema_alpha * cand.app_emb_ema
                + (1.0 - self.cfg.emb_ema_alpha) * e
            )

        cand.hits += 1
        cand.misses = 0
        cand.age += 1
        cand.last_seen_frame = int(frame_idx)

    def _append_matched_pending_debug(
        self,
        cand: PendingCandidate,
        det: Detection,
        match_row: Tuple[Any, ...],
        stable_tracks: Sequence[Track],
        debug: Dict[str, Any],
    ) -> None:
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
        ) = match_row

        progress = self._pending_progress_debug(cand)
        debug["candidate_events"].append({
            **self._pending_identity_debug(cand),
            "event": "matched_detection",
            "det_idx": int(j),
            "matched_det_idx": int(j),
            "status": cand.status,
            **progress,
            "ignore_overlap_on_birth": cand.birth_mode == "easy_start",
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
            "pending_id": cand.pending_id,
            "birth_mode": cand.birth_mode,
            "very_close_bypass": bool(cand.very_close_bypass),
            **progress,
            "easy_birth_created_count": int(self.easy_birth_created_count),
            "easy_birth_track_limit": int(self.cfg.easy_birth_track_limit),
            "pending_comparison": {
                "pending_id": cand.pending_id,
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

    def _append_missed_pending_debug(
        self,
        cand: PendingCandidate,
        pid: str,
        debug: Dict[str, Any],
    ) -> None:
        debug["candidate_events"].append({
            **self._pending_identity_debug(cand),
            "event": "missed_this_frame",
            "status": cand.status,
            **self._pending_progress_debug(cand),
            "ignore_overlap_on_birth": cand.birth_mode == "easy_start",
            "easy_birth_created_count": int(self.easy_birth_created_count),
            "easy_birth_track_limit": int(self.cfg.easy_birth_track_limit),
            **self._pending_frame_attempt_debug(
                pending_id=pid,
                comparisons=debug["pending_comparisons"],
            ),
        })

    def _refresh_blocked_candidate(
        self,
        cand: PendingCandidate,
        stable_tracks: Sequence[Track],
        debug: Dict[str, Any],
    ) -> None:
        if cand.status != "blocked":
            return

        nearest = self._nearest_existing(cand.last_detection, stable_tracks)
        nearest_id = nearest["nearest_existing_track_id"]
        nearest_d = float(nearest["nearest_existing_d_motion"])

        cand.nearest_existing_track_id = nearest_id
        cand.nearest_existing_d_motion = nearest_d

        if nearest_d <= self.cfg.very_close_threshold:
            return

        old_status = cand.status
        cand.status = "near_existing" if nearest_d <= self.cfg.near_threshold else "normal"

        if cand.status != old_status:
            debug["candidate_events"].append({
                **self._pending_identity_debug(cand),
                "event": "status_changed",
                "from_status": old_status,
                "to_status": cand.status,
                "nearest_existing_track_id": nearest_id,
                "nearest_existing_d_motion": nearest_d,
                **self._pending_progress_debug(cand),
            })

    def _queue_confirmed_if_ready(
        self,
        cand: PendingCandidate,
        debug: Dict[str, Any],
    ) -> None:
        if cand.hits < cand.required_confirm_hits or cand.age > self.cfg.max_pending_age:
            return

        progress = self._pending_progress_debug(cand)
        debug["confirmed"].append({
            **self._pending_identity_debug(cand),
            "source_det_idx": int(cand.last_det_idx),
            "reason": "stable_pending_candidate",
            "status": cand.status,
            **progress,
            "hits_left_to_track": 0,
            "ready_for_track": True,
            "ignore_overlap_on_birth": cand.birth_mode == "easy_start",
            "easy_birth_created_count": int(self.easy_birth_created_count),
            "easy_birth_track_limit": int(self.cfg.easy_birth_track_limit),
            "will_create_new_track": True,
        })

    def _expire_pending_if_needed(
        self,
        pid: str,
        cand: PendingCandidate,
        debug: Dict[str, Any],
    ) -> None:
        if cand.age <= self.cfg.max_pending_age and cand.misses <= self.cfg.max_pending_misses:
            return

        debug["candidate_events"].append({
            **self._pending_identity_debug(cand),
            "event": "removed",
            "reason": "expired_pending_candidate",
            **self._pending_progress_debug(cand),
        })
        self.pending.pop(pid, None)

    def _update_existing_pending(
        self,
        matched: Dict[str, Tuple[Any, ...]],
        detections: Sequence[Detection],
        stable_tracks: Sequence[Track],
        frame_idx: int,
        debug: Dict[str, Any],
    ) -> None:
        for pid, cand in list(self.pending.items()):
            if pid in matched:
                match_row = matched[pid]
                det_idx = int(match_row[2])
                det = detections[det_idx]
                self._apply_pending_match(cand, det, det_idx, frame_idx)
                self._append_matched_pending_debug(cand, det, match_row, stable_tracks, debug)
            else:
                cand.misses += 1
                cand.age += 1
                self._append_missed_pending_debug(cand, pid, debug)

            self._refresh_blocked_candidate(cand, stable_tracks, debug)
            self._queue_confirmed_if_ready(cand, debug)
            self._expire_pending_if_needed(pid, cand, debug)

    def _commit_confirmed_candidates(self, debug: Dict[str, Any]) -> List[int]:
        confirmed = []
        confirmed_ids = {x["pending_id"] for x in debug["confirmed"]}

        for pid in confirmed_ids:
            cand = self.pending.pop(pid, None)
            if cand is None:
                continue

            birth_mode, ignore_overlap_on_birth = self._mark_confirmed_birth(cand)
            if birth_mode == "easy_start":
                self.easy_birth_created_count += 1

            for item in debug["confirmed"]:
                if item.get("pending_id") == pid:
                    item["birth_mode"] = birth_mode
                    item["ignore_overlap_on_birth"] = bool(ignore_overlap_on_birth)

            confirmed.append(int(cand.last_det_idx))

        return confirmed

    def _classify_new_pending(
        self,
        nearest_d: float,
        easy_birth_reserved_count: int,
    ) -> Tuple[str, str, str, int, str, bool, int]:
        easy_birth_available = (
            self.cfg.easy_birth_track_limit > 0
            and self.easy_birth_created_count + easy_birth_reserved_count
            < self.cfg.easy_birth_track_limit
        )

        if easy_birth_available:
            return (
                "normal",
                "pending_created",
                "easy_start",
                max(1, int(self.cfg.easy_birth_confirm_hits)),
                "easy_start",
                nearest_d <= self.cfg.very_close_threshold,
                easy_birth_reserved_count + 1,
            )

        if nearest_d <= self.cfg.very_close_threshold:
            return (
                "blocked",
                "blocked_pending",
                "very_close_to_existing_track",
                self.cfg.very_close_confirm_hits,
                "normal",
                False,
                easy_birth_reserved_count,
            )

        if nearest_d <= self.cfg.near_threshold:
            return (
                "near_existing",
                "pending_created",
                "near_existing_track_requires_more_confirmation",
                self.cfg.near_confirm_hits,
                "normal",
                False,
                easy_birth_reserved_count,
            )

        return (
            "normal",
            "pending_created",
            "far_from_existing_track",
            self.cfg.normal_confirm_hits,
            "normal",
            False,
            easy_birth_reserved_count,
        )

    def _make_pending_candidate(
        self,
        det: Detection,
        det_idx: int,
        frame_idx: int,
        status: str,
        nearest_id: Optional[int],
        nearest_d: float,
        required_hits: int,
        birth_mode: str,
        very_close_bypass: bool,
    ) -> PendingCandidate:
        pid = self._new_pending_id()
        kf = KalmanTracker(
            x0=[det.center[0], det.center[1], 0.0, 0.0],
            dt=self.dt,
            process_var=self.process_var,
            measure_var=self.measure_var,
            p0=self.p0,
        )
        emb = det.meta.get("e_app") if isinstance(det.meta, dict) else None

        return PendingCandidate(
            pid,
            kf,
            det,
            int(det_idx),
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
            required_confirm_hits=int(required_hits),
            birth_mode=birth_mode,
            very_close_bypass=very_close_bypass,
        )

    def _append_created_pending_debug(
        self,
        cand: PendingCandidate,
        det: Detection,
        det_idx: int,
        nearest: Dict[str, Any],
        action: str,
        reason: str,
        debug: Dict[str, Any],
    ) -> None:
        progress = self._pending_progress_debug(cand)
        debug["candidate_events"].append({
            **self._pending_identity_debug(cand),
            "event": "created",
            "det_idx": int(det_idx),
            "status": cand.status,
            **progress,
            "ignore_overlap_on_birth": cand.birth_mode == "easy_start",
            "easy_birth_created_count": int(self.easy_birth_created_count),
            "easy_birth_track_limit": int(self.cfg.easy_birth_track_limit),
            "nearest_existing_track_id": nearest["nearest_existing_track_id"],
            "nearest_existing_d_motion": float(nearest["nearest_existing_d_motion"]),
        })

        debug["detections"].append({
            "det_idx": int(det_idx),
            "det_center": tuple(map(float, det.center)),
            "action": action,
            "reason": reason,
            **nearest,
            "closeness_status": self._closeness_status(float(nearest["nearest_existing_d_motion"])),
            "very_close_threshold": float(self.cfg.very_close_threshold),
            "near_threshold": float(self.cfg.near_threshold),
            "normal_confirm_hits": int(self.cfg.normal_confirm_hits),
            "near_confirm_hits": int(self.cfg.near_confirm_hits),
            "very_close_confirm_hits": int(self.cfg.very_close_confirm_hits),
            "pending_id": cand.pending_id,
            "birth_mode": cand.birth_mode,
            "very_close_bypass": bool(cand.very_close_bypass),
            **progress,
            "easy_birth_created_count": int(self.easy_birth_created_count),
            "easy_birth_track_limit": int(self.cfg.easy_birth_track_limit),
            "will_create_new_track": False,
        })

    def _confirm_easy_start_birth(
        self,
        cand: PendingCandidate,
        confirmed: List[int],
        debug: Dict[str, Any],
    ) -> bool:
        if cand.birth_mode != "easy_start" or cand.hits < cand.required_confirm_hits:
            return False

        self.pending.pop(cand.pending_id, None)
        birth_mode, ignore_overlap_on_birth = self._mark_confirmed_birth(cand)
        self.easy_birth_created_count += 1
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
        return True

    def _create_new_pending_from_unmatched(
        self,
        unmatched_det_indices: Sequence[int],
        detections: Sequence[Detection],
        stable_tracks: Sequence[Track],
        taken_d: set,
        frame_idx: int,
        confirmed: List[int],
        debug: Dict[str, Any],
    ) -> None:
        easy_birth_reserved_count = sum(
            1 for cand in self.pending.values()
            if cand.birth_mode == "easy_start"
        )

        for j in unmatched_det_indices:
            if int(j) in taken_d:
                continue

            det = detections[int(j)]
            nearest = self._nearest_existing(det, stable_tracks)
            nearest_id = nearest["nearest_existing_track_id"]
            nearest_d = float(nearest["nearest_existing_d_motion"])

            (
                status,
                action,
                reason,
                required_hits,
                birth_mode,
                very_close_bypass,
                easy_birth_reserved_count,
            ) = self._classify_new_pending(nearest_d, easy_birth_reserved_count)

            cand = self._make_pending_candidate(
                det=det,
                det_idx=int(j),
                frame_idx=frame_idx,
                status=status,
                nearest_id=nearest_id,
                nearest_d=nearest_d,
                required_hits=required_hits,
                birth_mode=birth_mode,
                very_close_bypass=very_close_bypass,
            )
            self.pending[cand.pending_id] = cand
            self._append_created_pending_debug(cand, det, int(j), nearest, action, reason, debug)

            if self._confirm_easy_start_birth(cand, confirmed, debug):
                easy_birth_reserved_count -= 1

    def _attach_pending_snapshot(self, debug: Dict[str, Any]) -> None:
        debug["candidates"] = [
            {
                **self._pending_identity_debug(cand),
                **self._pending_progress_debug(cand),
                "birth_mode": cand.birth_mode,
                "very_close_bypass": bool(cand.very_close_bypass),
                "last_det_idx": int(cand.last_det_idx),
                "last_center": tuple(map(float, cand.last_center)),
                "nearest_existing_track_id": cand.nearest_existing_track_id,
                "nearest_existing_d_motion": cand.nearest_existing_d_motion,
            }
            for cand in self.pending.values()
        ]

    def _finalize_debug(self, debug: Dict[str, Any], confirmed: Sequence[int]) -> None:
        debug["summary"]["confirmed_count"] = int(len(confirmed))
        debug["summary"]["pending_count_after"] = int(len(self.pending))
        debug["summary"]["easy_birth_created_count"] = int(self.easy_birth_created_count)

    def update(
        self,
        unmatched_det_indices: Sequence[int],
        detections: Sequence[Detection],
        existing_tracks: Sequence[Track],
        frame_idx: int,
        g: float = 1.0,
    ) -> BirthResult:
        stable_tracks = self._stable_tracks(existing_tracks)
        debug = self._build_update_debug(
            unmatched_det_indices,
            existing_tracks,
            stable_tracks,
            frame_idx,
        )

        self._predict_pending()
        pair_rows = self._build_pending_pair_rows(unmatched_det_indices, detections, debug)
        taken_d, matched = self._select_pending_matches(pair_rows, debug)
        self._update_existing_pending(matched, detections, stable_tracks, frame_idx, debug)

        confirmed = self._commit_confirmed_candidates(debug)
        self._create_new_pending_from_unmatched(
            unmatched_det_indices=unmatched_det_indices,
            detections=detections,
            stable_tracks=stable_tracks,
            taken_d=taken_d,
            frame_idx=frame_idx,
            confirmed=confirmed,
            debug=debug,
        )

        self._attach_pending_snapshot(debug)
        self._finalize_debug(debug, confirmed)

        return BirthResult(
            confirmed_birth_det_indices=sorted(set(confirmed)),
            debug_info=debug,
        )
