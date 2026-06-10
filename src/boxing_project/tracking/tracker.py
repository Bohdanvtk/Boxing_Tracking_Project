from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import math

from boxing_project.kalman_filter.kalman import KalmanTracker
from .track import Track, Detection
from .matcher import MatchConfig, match_tracks_and_detections
from .birth_manager import BirthManager
from .overlap_manager import OverlapManager
from .tracking_reporter import (
    attach_birth_debug,
    attach_freeze_debug,
    attach_removed_tracks_debug,
    attach_track_update_debug,
    build_active_tracks_summary,
    build_track_update_record,
    build_used_config_debug,
    write_app_decision_meta,
    write_match_meta,
    write_update_gate_meta,
)
from . import DEFAULT_TRACKING_CONFIG_PATH, DEFAULT_BIRTH_CONFIG_PATH


@dataclass
class TrackerConfig:
    dt: float
    process_var: float
    measure_var: float
    p0: float
    max_age: int
    max_confirmed_age: int
    min_hits: int
    min_hits_sub: int
    match: MatchConfig
    min_kp_conf: float
    reset_g_threshold: float
    debug: bool
    save_log: bool
    max_unconfirmed_tracks: int = 6

    overlap_log_threshold: float = 0.10
    skeleton_overlap_threshold: float = 0.08
    skeleton_overlap_full_weight: float = 0.35
    skeleton_overlap_core_weight: float = 0.65
    skeleton_overlap_conf_threshold: float = 0.05
    skeleton_overlap_thickness: int = 7
    # Center-distance adaptive overlap gating.
    adaptive_overlap_center_near: float = 0.55
    adaptive_overlap_center_mid: float = 0.85
    adaptive_overlap_center_far: float = 1.20
    adaptive_overlap_iou_near: float = 0.03
    adaptive_overlap_iou_mid: float = 0.06
    adaptive_overlap_iou_far: float = 0.08
    adaptive_overlap_iou_default: float = 0.12
    overlap_app_freeze_after: int = 5
    # Configurable overlap-safety weak motion update. Clamped at use site.
    overlap_motion_alpha: float = 0.25
    w_body: float = 1.0
    w_left_glove: float = 0.5
    w_right_glove: float = 0.5
    w_shorts: float = 0.75
    graph_clustering: Dict[str, Any] = field(default_factory=dict)


def openpose_people_to_detections(
    people: List[Dict[str, Any]],
    min_kp_conf: float = 0.05,
) -> List[Detection]:
    dets: List[Detection] = []
    for person in people:
        kps: Optional[np.ndarray] = None

        if 'pose_keypoints_2d' in person and isinstance(person['pose_keypoints_2d'], (list, tuple)):
            arr = np.asarray(person['pose_keypoints_2d'], dtype=float).reshape(-1)
            if arr.size % 3 != 0:
                continue
            K = arr.size // 3
            kps = arr.reshape(K, 3)

        elif 'keypoints' in person:
            arr = np.asarray(person['keypoints'], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                if arr.shape[1] == 2:
                    arr = np.concatenate([arr, np.ones((arr.shape[0], 1), dtype=float)], axis=1)
                kps = arr[:, :3]

        elif 'pose' in person:
            arr = np.asarray(person['pose'], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                if arr.shape[1] == 2:
                    arr = np.concatenate([arr, np.ones((arr.shape[0], 1), dtype=float)], axis=1)
                kps = arr[:, :3]

        elif 'pose_2d' in person:
            p = person['pose_2d']
            xs = np.asarray(p.get('x', []), dtype=float).reshape(-1, 1)
            ys = np.asarray(p.get('y', []), dtype=float).reshape(-1, 1)
            cs = np.asarray(p.get('conf', []), dtype=float).reshape(-1, 1)
            if xs.shape == ys.shape == cs.shape and xs.size > 0:
                kps = np.concatenate([xs, ys, cs], axis=1)

        if kps is None:
            continue

        good = kps[:, 2] >= float(min_kp_conf)
        xy = kps[:, :2].copy()
        xy[~good] = np.nan
        if np.all(~good):
            continue

        TORSO = [1, 8]  # Neck, MidHip (BODY_25)
        txy = xy[TORSO]
        if np.all(np.isnan(txy)):
            txy = xy
        cx, cy = np.nanmedian(txy, axis=0)

        dets.append(
            Detection(
                center=(float(cx), float(cy)),
                keypoints=xy,
                kp_conf=kps[:, 2],
                meta={'raw': person}
            )
        )

    return dets


@lru_cache(maxsize=None)
def _cached_tracking_config(path: str):
    from boxing_project.utils.config import load_tracking_config
    return load_tracking_config(path)


def _load_tracker_config_from_yaml(
    config_path: Optional[Union[str, Path]] = None,
) -> Tuple[TrackerConfig, Dict[str, Any]]:
    resolved = Path(config_path) if config_path is not None else DEFAULT_TRACKING_CONFIG_PATH
    tracker_cfg, match_cfg, raw_cfg = _cached_tracking_config(str(resolved))
    tracker_cfg_copy = copy.deepcopy(tracker_cfg)
    tracker_cfg_copy.match = copy.deepcopy(match_cfg)
    return tracker_cfg_copy, copy.deepcopy(raw_cfg)


class MultiObjectTracker:
    """Coordinates the tracking pipeline across frames.

    It owns global track state and delegates specialized logic to matcher,
    BirthManager, OverlapManager, and tracking_reporter.
    """

    def __init__(
        self,
        cfg: Optional[TrackerConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        if cfg is not None and config_path is not None:
            raise ValueError("Provide either cfg or config_path, not both")

        if cfg is None:
            cfg_loaded, raw_cfg = _load_tracker_config_from_yaml(config_path)
            self.cfg = cfg_loaded
            self._raw_config = raw_cfg
            self.config_path: Optional[Path] = (
                Path(config_path) if config_path is not None else DEFAULT_TRACKING_CONFIG_PATH
            )
        else:
            self.cfg = copy.deepcopy(cfg)
            self._raw_config = None
            self.config_path = Path(config_path) if config_path is not None else None

        self.tracks: List[Track] = []
        self._segment_tracks: Dict[int, Track] = {}
        self._next_id: int = 1
        self._was_reset_mode: bool = False
        self._epoch_id: int = 1
        self._epoch_tracks: Dict[int, Dict[int, Track]] = {self._epoch_id: {}}

        # Previous frame matches: track_id -> det_idx.
        # Used to detect which previously matched tracks disappeared now.
        self.prev_matches: Dict[int, int] = {}
        self._frame_idx: int = 0

        raw = self.get_config_dict() or {}
        self.debug: bool = bool(raw.get("tracking", {}).get("debug", False))

        from boxing_project.utils.config import load_birth_config

        birth_cfg = load_birth_config(str(DEFAULT_BIRTH_CONFIG_PATH))

        if birth_cfg.chi2_gating <= 0:
            birth_cfg.chi2_gating = float(self.cfg.match.chi2_gating)
        if birth_cfg.emb_ema_alpha <= 0:
            birth_cfg.emb_ema_alpha = float(self.cfg.match.emb_ema_alpha)
        if birth_cfg.min_kp_conf <= 0:
            birth_cfg.min_kp_conf = float(self.cfg.match.min_kp_conf)
        if birth_cfg.min_core_kps <= 0:
            birth_cfg.min_core_kps = int(self.cfg.match.min_core_kps_create_track)

        self.birth_manager = BirthManager(
            cfg=birth_cfg,
            dt=float(self.cfg.dt),
            process_var=float(self.cfg.process_var),
            measure_var=float(self.cfg.measure_var),
            p0=float(self.cfg.p0),
            pose_core=list(self.cfg.match.pose_core),
        )

        self.overlap_manager = OverlapManager(self.cfg)

    def get_config_dict(self) -> Optional[Dict[str, Any]]:
        if self._raw_config is None:
            return None
        return copy.deepcopy(self._raw_config)

    def _new_track(self, det: Detection) -> Track:
        kf = KalmanTracker(
            x0=[det.center[0], det.center[1], 0.0, 0.0],
            dt=self.cfg.dt,
            process_var=self.cfg.process_var,
            measure_var=self.cfg.measure_var,
            p0=self.cfg.p0,
        )

        trk = Track(
            track_id=self._next_id,
            kf=kf,
            min_hits=self.cfg.min_hits,
            min_hits_sub=self.cfg.min_hits_sub,
            epoch_id=self._epoch_id,
        )

        self._next_id += 1
        self._segment_tracks[trk.track_id] = trk
        self._epoch_tracks.setdefault(self._epoch_id, {})[trk.track_id] = trk

        ignore_overlap_on_birth = (
            det.meta.get("birth_mode") == "easy_start"
            and bool(det.meta.get("ignore_overlap_on_birth", False))
        )

        det.meta["ignore_overlap_on_birth"] = bool(ignore_overlap_on_birth)
        det.meta["birth_base_kps_bypass"] = bool(ignore_overlap_on_birth)

        # New tracks do not have current-frame match context here.
        # Therefore adaptive overlap normally falls back to default.
        has_overlap = self.overlap_manager.prepare_overlap_update_meta(trk, det)

        # Configurable create-track threshold controls whether a new track may
        # initialize/update appearance from this detection.
        trk.update(
            det,
            ema_alpha=self.cfg.match.emb_ema_alpha,
            update_app=(
                bool(ignore_overlap_on_birth)
                or self._has_base_keypoints(det, self.cfg.match.min_core_kps_create_track)
            ),
            has_overlap=has_overlap,
            ignore_overlap=ignore_overlap_on_birth,
            overlap_motion_alpha=self.cfg.overlap_motion_alpha,
        )

        return trk

    # ------------------------------------------------------------------
    # Appearance EMA recovery buffer logic
    # ------------------------------------------------------------------
    def _decide_app_buffer_update(
        self,
        trk: Track,
        det: Detection,
        d_motion: float,
        d_pose: float,
        d_app: float,
        max_update_app: float,
        max_update_motion: float,
        max_update_pose: float,
        has_overlap: bool,
        freeze_active: bool,
        det_idx: int,
    ) -> Dict[str, Any]:
        """
        Decide only appearance-memory update behavior for an already matched pair.

        This method does NOT decide matching. Track<->detection matching has already
        been resolved by the matcher. Here we only decide how (or whether) the
        matched detection is allowed to affect track appearance memory.

        Modes:
          - "strict":
              direct EMA update path via Track.update(update_app=True).
          - "buffer_candidate":
              detection embedding is buffered for recovery, but main EMA is not
              updated yet on this frame.
          - "recovery_batch_update":
              recovery buffer reached min size and averaged embedding was applied
              through weak EMA update.
          - "reject":
              detection is not allowed to update appearance memory or buffer.

        Strict update requires:
          d_app <= max_update_app, no overlap, no freeze, and e_app exists.

        Recovery candidate requires:
          d_app above strict threshold but below adaptive buffer upper threshold,
          plus safe motion/pose/coverage/overlap/freeze gates.

        Adaptive relaxation:
          buffer_upper_eff grows from app_buffer_upper toward app_buffer_hard_upper
          as app_stale_frames increases.

        Reject path:
          unsafe detections do not update EMA; strong reject reasons can clear
          recovery buffer according to config clear-policy flags.

        Returned payload is later copied into det.meta and track_update_debug.
        """
        # Read appearance input and current buffer state.
        e_app = det.meta.get("e_app", None)
        app_coverage = float(det.meta.get("match_app_coverage", det.meta.get("e_app_coverage", 1.0)))
        strict_app_update = d_app <= max_update_app and (not has_overlap) and (not freeze_active) and (e_app is not None)
        stale_before = int(trk.app_stale_frames)
        buffer_size_before = int(trk.get_app_buffer_size())
        clear_reason = None
        reject_reason = None
        recovery_batch_applied = False

        # Compute adaptive buffer upper threshold.
        app_buffer_upper = float(getattr(self.cfg.match, "app_buffer_upper", 0.12))
        app_buffer_hard_upper = float(getattr(self.cfg.match, "app_buffer_hard_upper", 0.18))
        app_buffer_relax_tau = max(float(getattr(self.cfg.match, "app_buffer_relax_tau", 8.0)), 1e-6)
        relax = 1.0 - math.exp(-float(stale_before) / app_buffer_relax_tau)
        buffer_upper_eff = min(app_buffer_upper + (app_buffer_hard_upper - app_buffer_upper) * relax, app_buffer_hard_upper)

        # Safety gates for recovery buffer entry.
        motion_ok = d_motion <= float(getattr(self.cfg.match, "app_buffer_max_motion", max_update_motion))
        pose_ok = d_pose <= float(getattr(self.cfg.match, "app_buffer_max_pose", max_update_pose))
        coverage_ok = app_coverage >= float(getattr(self.cfg.match, "app_buffer_min_coverage", 0.70))
        overlap_ok = not has_overlap
        freeze_ok = not freeze_active

        recovery_candidate = (
            (not strict_app_update) and (e_app is not None) and (d_app > max_update_app)
            and (d_app <= buffer_upper_eff) and (d_app <= app_buffer_hard_upper)
            and motion_ok and pose_ok and coverage_ok and overlap_ok and freeze_ok
        )

        mode = "reject"
        update_app = False
        # Mode 1: strict direct EMA update.
        if strict_app_update:
            mode = "strict"
            update_app = True
        # Mode 2: recovery buffer candidate.
        elif recovery_candidate:
            mode = "buffer_candidate"
            trk.add_app_buffer_embedding(e_app=e_app, meta={"frame_idx": self._frame_idx, "det_idx": int(det_idx)})
            if trk.get_app_buffer_size() >= max(1, int(getattr(self.cfg.match, "app_buffer_min_size", 3))):
                batch_emb = np.mean(np.stack(trk.app_update_buffer, axis=0), axis=0)
                if trk.apply_recovery_batch_update(batch_emb, float(getattr(self.cfg.match, "app_buffer_recovery_ema_alpha", 0.97))):
                    mode = "recovery_batch_update"
                    recovery_batch_applied = True
                    trk.app_stale_frames = 0
                    trk.clear_app_buffer("recovery_batch_update")
                    clear_reason = "recovery_batch_update"
        # Mode 3: reject and optional buffer clear.
        else:
            hard_reject = d_app > app_buffer_hard_upper
            safety_fail = not (motion_ok and pose_ok and coverage_ok)
            if has_overlap:
                reject_reason = "overlap"
            elif freeze_active:
                reject_reason = "freeze"
            elif e_app is None:
                reject_reason = "missing_e_app"
            elif hard_reject:
                reject_reason = "hard_reject"
            elif safety_fail:
                reject_reason = "safety_fail"
            else:
                reject_reason = "above_buffer_upper"
            if has_overlap and bool(getattr(self.cfg.match, "app_buffer_clear_on_overlap", True)):
                trk.clear_app_buffer("overlap")
                clear_reason = "overlap"
            elif freeze_active and bool(getattr(self.cfg.match, "app_buffer_clear_on_freeze", True)):
                trk.clear_app_buffer("freeze")
                clear_reason = "freeze"
            elif hard_reject and bool(getattr(self.cfg.match, "app_buffer_clear_on_hard_reject", True)):
                trk.clear_app_buffer("hard_reject")
                clear_reason = "hard_reject"
            elif safety_fail and bool(getattr(self.cfg.match, "app_buffer_clear_on_safety_fail", True)):
                trk.clear_app_buffer("safety_fail")
                clear_reason = "safety_fail"

        # Return decision/debug payload.
        return {
            "update_app": bool(update_app),
            "mode": mode,
            "strict": bool(strict_app_update),
            "recovery_candidate": bool(recovery_candidate),
            "recovery_batch_applied": bool(recovery_batch_applied),
            "coverage": float(app_coverage),
            "buffer_upper_eff": float(buffer_upper_eff),
            "buffer_base_upper": float(app_buffer_upper),
            "buffer_hard_upper": float(app_buffer_hard_upper),
            "stale_before": int(stale_before),
            "buffer_size_before": int(buffer_size_before),
            "clear_reason": clear_reason,
            "reject_reason": reject_reason,
            "motion_ok": bool(motion_ok),
            "pose_ok": bool(pose_ok),
            "coverage_ok": bool(coverage_ok),
            "overlap_ok": bool(overlap_ok),
            "freeze_ok": bool(freeze_ok),
        }

    def _prune_unconfirmed_tracks(self) -> List[Dict[str, Any]]:
        """
        Keep all confirmed tracks and limit only older unconfirmed tracks.

        Important:
          - confirmed tracks are never pruned,
          - newly born tracks with age <= 1 are protected for one frame,
          - pruning is controlled by max_unconfirmed_tracks.
        """

        max_unconfirmed = max(0, int(getattr(self.cfg, "max_unconfirmed_tracks", 6)))

        protected_ids = {
            int(t.track_id)
            for t in self.tracks
            if not bool(getattr(t, "confirmed", False))
               and int(getattr(t, "age", 0)) <= 1
        }

        unconfirmed = [
            t for t in self.tracks
            if not bool(getattr(t, "confirmed", False))
               and not bool(getattr(t, "sub_confirmed", False))
               and int(t.track_id) not in protected_ids
        ]

        if len(unconfirmed) <= max_unconfirmed:
            return []

        keep = sorted(
            unconfirmed,
            key=lambda t: (
                int(getattr(t, "hits", 0)),
                -int(getattr(t, "time_since_update", 0)),
                -int(getattr(t, "age", 0)),
            ),
            reverse=True,
        )[:max_unconfirmed]

        keep_ids = {int(t.track_id) for t in keep} | protected_ids

        removed = [
            {
                "track_id": int(t.track_id),
                "hits": int(getattr(t, "hits", 0)),
                "age": int(getattr(t, "age", 0)),
                "time_since_update": int(getattr(t, "time_since_update", 0)),
                "sub_confirmed": bool(getattr(t, "sub_confirmed", False)),
                "confirmed": bool(getattr(t, "confirmed", False)),
                "reason": "max_unconfirmed_tracks",
            }
            for t in self.tracks
            if not bool(getattr(t, "confirmed", False))
               and not bool(getattr(t, "sub_confirmed", False))
               and int(t.track_id) not in keep_ids
        ]

        self.tracks = [
            t for t in self.tracks
            if bool(getattr(t, "confirmed", False))
               or bool(getattr(t, "sub_confirmed", False))
               or int(t.track_id) in keep_ids
        ]

        for item in removed:
            self.prev_matches.pop(int(item["track_id"]), None)

        return removed

    def _remove_dead(self) -> List[Dict[str, Any]]:
        removed = [
            {
                "track_id": int(t.track_id),
                "hits": int(getattr(t, "hits", 0)),
                "age": int(getattr(t, "age", 0)),
                "time_since_update": int(getattr(t, "time_since_update", 0)),
                "sub_confirmed": bool(getattr(t, "sub_confirmed", False)),
                "confirmed": bool(getattr(t, "confirmed", False)),
                "reason": (
                    "max_confirmed_age"
                    if bool(getattr(t, "confirmed", False))
                    else "max_age"
                ),
            }
            for t in self.tracks
            if t.is_dead(self.cfg.max_age, self.cfg.max_confirmed_age)
        ]

        self.tracks = [
            t for t in self.tracks
            if not t.is_dead(self.cfg.max_age, self.cfg.max_confirmed_age)
        ]

        for item in removed:
            self.prev_matches.pop(int(item["track_id"]), None)

        return removed
    def _has_base_keypoints(
        self,
        det: Detection,
        min_core_kps: Optional[int] = None,
    ) -> bool:
        """
        Check whether a detection contains enough valid keypoints
        from the configured core set.

        Strict mode:
            min_core_kps is None -> require all configured core keypoints.

        Relaxed mode:
            min_core_kps is int -> require at least this many valid core keypoints.
        """
        core = self.cfg.match.pose_core

        if not core:
            return True

        if det.keypoints is None:
            return False

        kps = np.asarray(det.keypoints, dtype=float)

        if kps.ndim != 2 or kps.shape[1] < 2:
            return False

        core = np.asarray(core, dtype=int).ravel()
        core = core[(0 <= core) & (core < kps.shape[0])]

        if core.size == 0:
            return False

        valid_count = int(np.isfinite(kps[core, :2]).all(axis=1).sum())
        required = int(core.size) if min_core_kps is None else int(min_core_kps)

        return valid_count >= required

    def update(
        self,
        detections: List[Detection],
        reset_mode: bool,
        g: float = 1.0,
    ) -> Dict[str, Any]:
        self._frame_idx += 1

        # Reset only on the reset edge.
        if reset_mode and not self._was_reset_mode:
            self.reset()

        # 1. Predict.
        for trk in self.tracks:
            trk.predict()

        # Matcher rows -> track ids.
        idx2tid = {
            i: t.track_id
            for i, t in enumerate(self.tracks)
        }

        row_track_ids = [
            idx2tid[i]
            for i in range(len(self.tracks))
        ]

        # 2. Match. Freeze does not block matching.
        matches_idx, um_tr_idx, um_det_idx, C, log = match_tracks_and_detections(
            tracks=self.tracks,
            detections=detections,
            cfg=self.cfg.match,
            debug=self.debug,
            g=g,
            reset_mode=reset_mode,
        )

        # track_id -> det_idx
        current_matches: Dict[int, int] = {
            int(self.tracks[i_track].track_id): int(j_det)
            for i_track, j_det in matches_idx
        }

        # det_idx -> Track for adaptive overlap.
        det_idx_to_track = self.overlap_manager.build_det_idx_to_track(self.tracks, matches_idx)

        matched_track_ids = set(current_matches.keys())

        # 3. Clear freeze sources that returned.
        self.overlap_manager.which_defroze(
            matched_track_ids=matched_track_ids,
            tracks=self.tracks,
        )

        # 4. Find tracks missing since previous frame.
        absent_ids = self.overlap_manager.compare_matches(
            prev_match=self.prev_matches,
            current_match=current_matches,
        )

        # 5. Freeze tracks near disappeared overlap sources.
        freshly_frozen_sources = self.overlap_manager.freeze_track_near_unmatched(
            absent_ids=absent_ids,
            tracks=self.tracks,
        )

        # 6. Update matched tracks.
        id_pairs: List[Tuple[int, int]] = []
        skipped_updates: List[Dict[str, Any]] = []

        max_update_cost = float(getattr(self.cfg.match, "max_update_cost", 1.2))
        max_update_motion = float(self.cfg.match.max_update_motion)
        max_update_pose = float(self.cfg.match.max_update_pose)
        max_update_app = float(self.cfg.match.max_update_app)

        track_update_debug: List[Dict[str, Any]] = []

        # Matrices come from matcher debug meta.
        motion_matrix = log.meta.get("motion_matrix", None)
        pose_matrix = log.meta.get("pose_matrix", None)
        app_matrix = log.meta.get("app_matrix", None)
        update_cost_matrix = log.meta.get("update_cost_matrix", None)

        if (
            motion_matrix is None
            or pose_matrix is None
            or app_matrix is None
            or update_cost_matrix is None
        ):
            raise RuntimeError(
                "Missing motion_matrix / pose_matrix / app_matrix / update_cost_matrix in log.meta. "
                "Check matcher.py build_cost_matrix()."
            )

        for i_track, j_det in matches_idx:
            trk = self.tracks[i_track]
            det = detections[j_det]

            i = int(i_track)
            j = int(j_det)

            d_motion = float(motion_matrix[i, j])
            d_pose = float(pose_matrix[i, j])
            d_app = float(app_matrix[i, j])

            row_cost = float(C[i, j])
            update_cost = float(update_cost_matrix[i, j])

            write_match_meta(
                det,
                trk,
                update_cost=update_cost,
                row_cost=row_cost,
                d_motion=d_motion,
                d_pose=d_pose,
                d_app=d_app,
                max_update_cost=max_update_cost,
            )

            id_pairs.append((trk.track_id, j_det))

            update_motion = d_motion <= max_update_motion
            update_pose = (
                d_pose <= max_update_pose
                and self._has_base_keypoints(det, self.cfg.match.min_core_kps_update)
            )
            has_overlap = self.overlap_manager.prepare_overlap_update_meta(
                trk=trk,
                det=det,
                det_idx_to_track=det_idx_to_track,
            )
            freeze_active = trk.is_frozen()
            app_decision = self._decide_app_buffer_update(
                trk=trk,
                det=det,
                d_motion=d_motion,
                d_pose=d_pose,
                d_app=d_app,
                max_update_app=max_update_app,
                max_update_motion=max_update_motion,
                max_update_pose=max_update_pose,
                has_overlap=has_overlap,
                freeze_active=freeze_active,
                det_idx=j_det,
            )
            update_app_mode = str(app_decision["mode"])
            update_app = bool(app_decision["update_app"])

            disabled_reasons = []

            if not update_motion:
                disabled_reasons.append("motion_update_disabled")
            if not update_pose:
                disabled_reasons.append("pose_update_disabled")
            if not update_app:
                disabled_reasons.append("app_update_disabled")

            write_update_gate_meta(det, disabled_reasons)

            trk.update(
                det,
                ema_alpha=self.cfg.match.emb_ema_alpha,
                update_motion=update_motion,
                update_pose=update_pose,
                update_app=update_app,
                has_overlap=has_overlap,
                overlap_motion_alpha=self.cfg.overlap_motion_alpha,
            )
            # Update appearance recovery counters after Track.update().
            if update_app_mode == "strict" and bool(det.meta.get("track_app_update_allowed", False)):
                if bool(getattr(self.cfg.match, "app_buffer_clear_on_strict_update", True)):
                    trk.clear_app_buffer("strict_update")
                    app_decision["clear_reason"] = "strict_update"
                trk.app_stale_frames = 0
            elif not bool(app_decision["recovery_batch_applied"]):
                trk.app_stale_frames += 1
            trk.last_app_update_mode = update_app_mode
            app_decision["stale_after"] = int(trk.app_stale_frames)
            app_decision["buffer_size_after"] = int(trk.get_app_buffer_size())
            write_app_decision_meta(det, app_decision, self.cfg)

            rec = build_track_update_record(
                track_idx=i,
                det_idx=j,
                trk=trk,
                det=det,
                d_motion=d_motion,
                d_pose=d_pose,
                d_app=d_app,
                row_cost=row_cost,
                update_cost=update_cost,
                max_update_cost=max_update_cost,
                max_update_motion=max_update_motion,
                max_update_pose=max_update_pose,
                max_update_app=max_update_app,
                update_motion=update_motion,
                update_pose=update_pose,
                update_app=update_app,
            )

            track_update_debug.append(rec)

            if rec["track_update_skipped"]:
                skipped_updates.append(rec)

        attach_track_update_debug(
            log,
            build_used_config_debug(self.cfg, self.birth_manager),
            track_update_debug,
        )

        # 7. Birth manager handles unmatched detections.
        birth_result = self.birth_manager.update(
            unmatched_det_indices=um_det_idx,
            detections=detections,
            existing_tracks=self.tracks,
            frame_idx=self._frame_idx,
            g=g,
        )

        attach_birth_debug(log, birth_result.debug_info)

        # 8. Spawn confirmed births.
        new_matches: Dict[int, int] = {}

        for j in birth_result.confirmed_birth_det_indices:
            trk = self._new_track(detections[j])
            self.tracks.append(trk)
            new_matches[int(trk.track_id)] = int(j)

        pruned_unconfirmed_tracks = self._prune_unconfirmed_tracks()
        pruned_ids = {
            int(item["track_id"])
            for item in pruned_unconfirmed_tracks
        }
        new_matches = {
            int(track_id): int(det_idx)
            for track_id, det_idx in new_matches.items()
            if int(track_id) not in pruned_ids
        }
        log.meta["pruned_unconfirmed_tracks"] = pruned_unconfirmed_tracks

        # 9. Include new tracks in current matches.
        all_current_matches: Dict[int, int] = {
            **current_matches,
            **new_matches,
        }

        # 10. Convert detection-overlap groups to track-id groups.
        track_groups = self.overlap_manager.dets_to_track(
            matches=all_current_matches,
            detections=detections,
        )

        # 11. Store groups for next-frame freeze logic.
        tracks_by_id = {
            int(track.track_id): track
            for track in self.tracks
        }

        for track_id, group_ids in track_groups.items():
            track = tracks_by_id.get(int(track_id))

            if track is not None:
                track.overlap_group_ids = {
                    int(group_id)
                    for group_id in group_ids
                }

        # Clear stale groups.
        for track in self.tracks:
            if int(track.track_id) not in track_groups:
                track.overlap_group_ids = set()

        # 12. Decrease freeze counters; skip newly frozen sources.
        self.overlap_manager.decrease_freeze(
            tracks=self.tracks,
            exclude_sources=freshly_frozen_sources,
        )

        attach_freeze_debug(log, self.tracks)

        # 13. Save matches for next frame.
        self.prev_matches = all_current_matches

        # 14. Cleanup.
        dead_tracks = self._remove_dead()
        attach_removed_tracks_debug(
            log,
            pruned=pruned_unconfirmed_tracks,
            dead=dead_tracks,
        )

        unmatched_track_ids = sorted(int(x) for x in absent_ids)
        active_tracks_summary = build_active_tracks_summary(
            self.tracks,
            max_age=self.cfg.max_age,
            max_confirmed_age=self.cfg.max_confirmed_age,
        )

        self._was_reset_mode = bool(reset_mode)

        return {
            "matches": id_pairs,
            "epoch_id": int(self._epoch_id),
            "unmatched_track_ids": unmatched_track_ids,
            "unmatched_det_indices": um_det_idx,
            "cost_matrix": C,
            "active_tracks": active_tracks_summary,
            "frame_log": log,
            "row_track_ids": row_track_ids,
            "skipped_updates": skipped_updates,
        }

    def get_active_tracks(self, confirmed_only: bool = True) -> List[Track]:
        if confirmed_only:
            return [
                t for t in self.tracks
                if t.confirmed and not t.is_dead(self.cfg.max_age, self.cfg.max_confirmed_age)
            ]

        return [
            t for t in self.tracks
            if not t.is_dead(self.cfg.max_age, self.cfg.max_confirmed_age)
        ]

    def get_segment_tracks(self) -> List[Track]:
        return list(self._segment_tracks.values())

    def reset(self):
        self.tracks.clear()
        self._segment_tracks.clear()
        self._next_id = 1

        self._epoch_id += 1
        self._epoch_tracks[self._epoch_id] = {}

        self.prev_matches.clear()
        self.birth_manager.reset()

    def get_epoch_tracks(self) -> Dict[int, Dict[int, Track]]:
        return {
            epoch_id: dict(tracks_by_id)
            for epoch_id, tracks_by_id in self._epoch_tracks.items()
        }
