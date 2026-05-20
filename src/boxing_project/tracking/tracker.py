from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import math

from boxing_project.kalman_filter.kalman import KalmanTracker
from .track import Track, Detection
from .matcher import MatchConfig, match_tracks_and_detections
from .birth_manager import BirthConfig, BirthManager
from .tracking_debug import (
    append_birth_debug,
    format_birth_debug_lines,
    format_freeze_debug_lines,
    format_track_update_debug_lines,
    format_removed_tracks_lines,
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
    overlap_mechanism: str = "skeleton_capsule"
    skeleton_overlap_threshold: float = 0.08
    skeleton_overlap_full_weight: float = 0.35
    skeleton_overlap_core_weight: float = 0.65
    skeleton_overlap_conf_threshold: float = 0.05
    skeleton_overlap_thickness: int = 7
    skeleton_overlap_relation_debug_mode: bool = True
    # Center-distance adaptive overlap gating.
    adaptive_overlap_center_near: float = 0.55
    adaptive_overlap_center_mid: float = 0.85
    adaptive_overlap_center_far: float = 1.20
    adaptive_overlap_iou_near: float = 0.03
    adaptive_overlap_iou_mid: float = 0.06
    adaptive_overlap_iou_far: float = 0.08
    adaptive_overlap_iou_default: float = 0.12
    overlap_app_freeze_after: int = 5
    w_body: float = 1.0
    w_left_glove: float = 0.5
    w_right_glove: float = 0.5
    w_shorts: float = 0.75


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
    """
    Multi-object tracker for boxing-player tracking.

    MultiObjectTracker owns the global tracking state across frames:
      - active Track objects,
      - track id allocation,
      - Kalman prediction/update scheduling,
      - matching tracks to detections,
      - spawning new tracks,
      - removing dead tracks,
      - reset/epoch bookkeeping,
      - and the overlap-group appearance cooldown system.

    Separation of responsibilities:
      Track is local.
      MultiObjectTracker is global.

    Track stores local state:
      - Kalman state,
      - appearance EMA,
      - overlap_group_ids,
      - freeze_sources.

    MultiObjectTracker decides global interactions:
      - which tracks matched this frame,
      - which tracks were matched last frame,
      - which tracks disappeared,
      - which disappeared tracks were in overlap groups,
      - which nearby tracks must freeze appearance memory,
      - which freeze sources should be cleared because the source track returned.

    Matching rule:
      Appearance cooldown never blocks matching.

      Tracks in cooldown still participate in match_tracks_and_detections().
      If a cooled-down track matches a detection, its Kalman/pose/hits update still runs.
      Cooldown blocks only app_emb_ema / appearance feature history updates.

    Overlap rule:
      Detection overlap is computed before tracker.update() and stored in det.meta:
        - max_overlap_iou
        - max_overlap_det_idx
        - overlap_relations
        - is_overlapping

      The tracker does not recompute IoU.
      It reads det.meta["overlap_relations"] and converts detection indices
      into track ids using current frame matches.

    Adaptive overlap rule:
      adaptive_overlap_enabled = True only when:
        1. current track is sub_confirmed or confirmed
        AND
        2. the current detection overlaps with at least one other detection
           that is matched to a sub_confirmed or confirmed track.

      Otherwise adaptive_overlap_iou_default is used.

      Examples:
        confirmed track + confirmed/sub_confirmed overlap -> adaptive
        confirmed track + pending/unmatched overlap -> default
        pending track + confirmed/sub_confirmed overlap -> default
        pending track + pending/unmatched overlap -> default

    prev_matches:
      Previous-frame matches:
          track_id -> det_idx

      Used by compare_matches() to detect which previously matched tracks
      disappeared on the current frame.

    overlap_group_ids:
      At the end of every frame, MultiObjectTracker builds current overlap groups.

      Detection overlap groups use det_idx.
      Tracks need track_id.
      dets_to_track() performs this conversion.

      Then every Track receives:
          track.overlap_group_ids = {track ids that overlapped with it now}

      On the next frame, if one of those ids disappears,
      nearby tracks can be frozen.

    Source-based cooldown:
      If Track M was in an overlap group on the previous frame and disappears now,
      then M becomes a freeze source.

      Every affected track receives:
          track.freeze_sources[M] = overlap_app_freeze_after

      Affected tracks include:
          - M itself, if still present in self.tracks,
          - every track whose overlap_group_ids contains M.

    Defreeze rule:
      A cooldown source is cleared only when that source track matches again.

      Example:
          N.freeze_sources = {M: 4}

      If N matches:
          nothing is cleared.

      If M matches:
          source M is cleared from every track:
              track.freeze_sources.pop(M, None)

      This is handled by which_defroze().

    Countdown rule:
      If a source track does not return, its cooldown decreases once per frame:
          freeze_sources[source] -= 1

      When it reaches zero, that source is removed.

      If no sources remain, appearance updates are allowed again,
      as long as the current detection has no dangerous overlap.

    High-level update() flow:
      1. Reset on reset edge if needed.
      2. Predict all existing tracks.
      3. Match tracks to detections.
      4. Build current_matches: track_id -> det_idx.
      5. Build det_idx_to_track: det_idx -> Track.
      6. which_defroze(): clear freeze sources for source tracks that returned.
      7. compare_matches(): find track ids that disappeared since prev_matches.
      8. freeze_track_near_unmatched(): start cooldown for tracks near disappeared sources.
      9. Update matched tracks.
      10. Spawn new tracks for unmatched detections.
      11. dets_to_track(): convert current detection-overlap groups into track-id groups.
      12. Write overlap_group_ids into Track objects for the next frame.
      13. decrease_freeze(): decrement active cooldown counters.
      14. Save prev_matches for the next frame.
      15. Remove dead tracks.

    Core idea:
      Motion/pose tracking should continue through overlap and occlusion.
      Appearance identity memory should be protected from contaminated crops.
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
            birth_cfg.min_core_kps = int(self.cfg.match.min_core_kps)

        self.birth_manager = BirthManager(
            cfg=birth_cfg,
            dt=float(self.cfg.dt),
            process_var=float(self.cfg.process_var),
            measure_var=float(self.cfg.measure_var),
            p0=float(self.cfg.p0),
            pose_core=list(self.cfg.match.pose_core),
        )

    def get_config_dict(self) -> Optional[Dict[str, Any]]:
        if self._raw_config is None:
            return None
        return copy.deepcopy(self._raw_config)

    def _build_det_idx_to_track(
        self,
        matches_idx: List[Tuple[int, int]],
    ) -> Dict[int, Track]:
        """
        Build reverse mapping for current-frame matches.

        matches_idx contains:
            (track_row_idx, det_idx)

        This returns:
            det_idx -> Track object

        This is needed because overlap metadata stores detection indices,
        but adaptive overlap logic needs to know which Track owns each detection.
        """
        return {
            int(j_det): self.tracks[int(i_track)]
            for i_track, j_det in matches_idx
        }

    def _track_stable(self, track: Optional[Track]) -> bool:
        return bool(track is not None and (track.sub_confirmed or track.confirmed))

    def _has_stable_overlap_counterpart(
        self,
        det: Detection,
        det_idx_to_track: Optional[Dict[int, Track]] = None,
    ) -> bool:
        if det_idx_to_track is None:
            return False
        return any(
            self._track_stable(det_idx_to_track.get(int(rel.get("det_idx"))))
            for rel in det.meta.get("overlap_relations", []) or []
            if rel.get("det_idx") is not None
        )

    def _pair_overlap_threshold(
        self,
        current_track: Track,
        other_track: Optional[Track],
        center_dist_norm: float,
    ) -> Tuple[float, str, str]:
        current_stable = self._track_stable(current_track)
        other_stable = self._track_stable(other_track)

        if not current_stable:
            return self.cfg.adaptive_overlap_iou_default, "default", "current_track_pending_default_only"
        if not other_stable:
            return self.cfg.adaptive_overlap_iou_default, "default", "other_track_pending_or_unmatched_default_only"
        if center_dist_norm <= self.cfg.adaptive_overlap_center_near:
            return self.cfg.adaptive_overlap_iou_near, "near", "stable_pair_adaptive"
        if center_dist_norm <= self.cfg.adaptive_overlap_center_mid:
            return self.cfg.adaptive_overlap_iou_mid, "mid", "stable_pair_adaptive"
        if center_dist_norm <= self.cfg.adaptive_overlap_center_far:
            return self.cfg.adaptive_overlap_iou_far, "far", "stable_pair_adaptive"
        return self.cfg.adaptive_overlap_iou_default, "default", "stable_pair_default_far"

    def _evaluate_detection_overlap(
        self,
        trk: Track,
        det: Detection,
        det_idx_to_track: Optional[Dict[int, Track]] = None,
    ) -> Dict[str, Any]:
        relations = det.meta.get("overlap_relations", []) or []
        risky_indices: List[int] = []
        risky_ious: List[float] = []
        best_any_rel: Optional[Dict[str, Any]] = None
        best_risky_rel: Optional[Dict[str, Any]] = None
        overlap_has_stable_track = False

        for rel in relations:
            other_idx = rel.get("det_idx")
            other_track = det_idx_to_track.get(int(other_idx)) if det_idx_to_track is not None and other_idx is not None else None
            other_stable = self._track_stable(other_track)
            overlap_has_stable_track = overlap_has_stable_track or other_stable
            cdn = float(rel.get("center_dist_norm", det.meta.get("min_center_dist_norm", float("inf"))))
            threshold, zone, reason = self._pair_overlap_threshold(trk, other_track, cdn)
            iou = float(rel.get("iou", 0.0))
            risky = bool(iou > float(threshold))

            rel["adaptive_overlap_threshold"] = float(threshold)
            rel["adaptive_overlap_zone"] = zone
            rel["adaptive_overlap_reason"] = reason
            rel["adaptive_overlap_risk"] = risky
            rel["other_track_stable"] = bool(other_stable)

            if best_any_rel is None or iou > float(best_any_rel.get("iou", 0.0)):
                best_any_rel = rel
            if risky:
                if best_risky_rel is None or iou > float(best_risky_rel.get("iou", 0.0)):
                    best_risky_rel = rel
                if other_idx is not None:
                    risky_indices.append(int(other_idx))
                risky_ious.append(iou)

        fallback = {
            "adaptive_overlap_threshold": float(self.cfg.adaptive_overlap_iou_default),
            "adaptive_overlap_zone": "default",
            "adaptive_overlap_reason": "no_overlap_relations_default_only",
        }
        active = best_risky_rel or best_any_rel or fallback
        return {
            "has_overlap": bool(risky_ious),
            "current_track_stable": self._track_stable(trk),
            "overlap_has_stable_track": bool(overlap_has_stable_track),
            "active_overlap_threshold": float(active.get("adaptive_overlap_threshold", self.cfg.adaptive_overlap_iou_default)),
            "adaptive_overlap_zone": active.get("adaptive_overlap_zone", "default"),
            "adaptive_overlap_reason": active.get("adaptive_overlap_reason", "no_overlap_relations_default_only"),
            "adaptive_overlap_enabled": str(active.get("adaptive_overlap_reason", "")).startswith("stable_pair"),
            "risky_overlap_count": len(risky_indices),
            "risky_overlap_det_indices": risky_indices,
            "max_risky_overlap_iou": max(risky_ious) if risky_ious else 0.0,
        }

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
        has_overlap = self._prepare_overlap_update_meta(trk, det)

        trk.update(
            det,
            ema_alpha=self.cfg.match.emb_ema_alpha,
            update_app=bool(ignore_overlap_on_birth) or self._has_base_keypoints(det),
            has_overlap=has_overlap,
            ignore_overlap=ignore_overlap_on_birth,
        )

        return trk

    def _prepare_overlap_update_meta(
        self,
        trk: Track,
        det: Detection,
        det_idx_to_track: Optional[Dict[int, Track]] = None,
    ) -> bool:
        eval_meta = self._evaluate_detection_overlap(trk, det, det_idx_to_track)
        raw_max_overlap_iou = float(det.meta.get("max_overlap_iou", 0.0))
        has_overlap = bool(eval_meta["has_overlap"])

        det.meta["track_hits_before_update"] = int(trk.hits)
        det.meta["track_sub_confirmed_before_update"] = bool(trk.sub_confirmed)
        det.meta["track_confirmed_before_update"] = bool(trk.confirmed)
        det.meta["track_has_risky_overlap"] = has_overlap
        det.meta["is_overlapping"] = has_overlap
        det.meta["adaptive_overlap_enabled"] = bool(eval_meta["adaptive_overlap_enabled"])
        det.meta["adaptive_overlap_reason"] = eval_meta["adaptive_overlap_reason"]
        det.meta["adaptive_overlap_zone"] = eval_meta["adaptive_overlap_zone"]
        det.meta["active_overlap_threshold"] = float(eval_meta["active_overlap_threshold"])
        det.meta["risky_overlap_count"] = int(eval_meta["risky_overlap_count"])
        det.meta["risky_overlap_det_indices"] = list(eval_meta["risky_overlap_det_indices"])
        det.meta["max_risky_overlap_iou"] = float(eval_meta["max_risky_overlap_iou"])
        det.meta["raw_max_overlap_iou"] = raw_max_overlap_iou
        det.meta["max_overlap_iou"] = raw_max_overlap_iou
        det.meta["max_overlap_det_idx"] = det.meta.get("max_overlap_det_idx")
        det.meta["min_center_dist_norm"] = float(det.meta.get("min_center_dist_norm", float("inf")))
        det.meta["center_dist_norm_det_idx"] = det.meta.get("center_dist_norm_det_idx")
        det.meta["current_track_stable_for_overlap"] = bool(eval_meta["current_track_stable"])
        det.meta["overlap_has_stable_track"] = bool(eval_meta["overlap_has_stable_track"])
        return has_overlap

    def _used_config_debug(self) -> Dict[str, Any]:
        cfg = self.cfg
        used = {
            "tracking.min_hits": int(cfg.min_hits),
            "tracking.min_hits_sub": int(cfg.min_hits_sub),
            "tracking.max_age": int(cfg.max_age),
            "tracking.max_confirmed_age": int(cfg.max_confirmed_age),
            "tracking.tracker.max_unconfirmed_tracks": int(cfg.max_unconfirmed_tracks),
            "tracking.overlap_mechanism": str(cfg.overlap_mechanism),
            "tracking.overlap_log_threshold": float(cfg.overlap_log_threshold),
            "tracking.skeleton_overlap_threshold": float(cfg.skeleton_overlap_threshold),
            "tracking.skeleton_overlap_full_weight": float(cfg.skeleton_overlap_full_weight),
            "tracking.skeleton_overlap_core_weight": float(cfg.skeleton_overlap_core_weight),
            "tracking.skeleton_overlap_conf_threshold": float(cfg.skeleton_overlap_conf_threshold),
            "tracking.skeleton_overlap_thickness": int(cfg.skeleton_overlap_thickness),
            "tracking.skeleton_overlap_relation_debug_mode": bool(cfg.skeleton_overlap_relation_debug_mode),
            "tracking.adaptive_overlap_center_near": float(cfg.adaptive_overlap_center_near),
            "tracking.adaptive_overlap_center_mid": float(cfg.adaptive_overlap_center_mid),
            "tracking.adaptive_overlap_center_far": float(cfg.adaptive_overlap_center_far),
            "tracking.adaptive_overlap_iou_near": float(cfg.adaptive_overlap_iou_near),
            "tracking.adaptive_overlap_iou_mid": float(cfg.adaptive_overlap_iou_mid),
            "tracking.adaptive_overlap_iou_far": float(cfg.adaptive_overlap_iou_far),
            "tracking.adaptive_overlap_iou_default": float(cfg.adaptive_overlap_iou_default),
            "tracking.overlap_app_freeze_after": int(cfg.overlap_app_freeze_after),
        }

        for name, value in vars(cfg.match).items():
            used[f"match.{name}"] = value

        birth_cfg = getattr(self.birth_manager, "cfg", None)

        if birth_cfg is not None:
            for name, value in vars(birth_cfg).items():
                used[f"birth.{name}"] = value

        return used

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

    def compare_matches(
        self,
        prev_match: Dict[int, int],
        current_match: Dict[int, int],
    ) -> set[int]:
        # Tracks matched in the previous frame but not matched now.
        return set(prev_match.keys()) - set(current_match.keys())

    def dets_to_track(
        self,
        matches: Dict[int, int],
        detections: List[Detection],
    ) -> Dict[int, set[int]]:
        """
        Convert detection overlap groups into track-id overlap groups.

        det.meta["overlap_relations"] stores det_idx.
        For cooldown logic we need track_id.

        If an overlapped detection is not matched to any track, it is skipped.
        """
        det_to_track = {
            int(det_idx): int(track_id)
            for track_id, det_idx in matches.items()
        }

        track_groups: Dict[int, set[int]] = {}

        for track_id, det_idx in matches.items():
            track_id = int(track_id)
            det_idx = int(det_idx)

            if not (0 <= det_idx < len(detections)):
                track_groups[track_id] = set()
                continue

            det = detections[det_idx]
            overlap_det_indices = Track.overlap_group(det)

            group_track_ids: set[int] = set()

            for other_det_idx in overlap_det_indices:
                other_track_id = det_to_track.get(int(other_det_idx))

                if other_track_id is None:
                    continue

                if int(other_track_id) == track_id:
                    continue

                group_track_ids.add(int(other_track_id))

            track_groups[track_id] = group_track_ids

        return track_groups

    def whether_in_group(
        self,
        idx: int,
        groups: Dict[int, set[int]],
    ) -> set[int]:
        # Return all track_ids that are in the same overlap group as idx.
        idx = int(idx)
        result: set[int] = set()

        for owner_id, members in groups.items():
            full_group = {int(owner_id)} | {int(x) for x in members}

            if idx in full_group:
                result |= full_group

        result.discard(idx)
        return result

    def set_cooldown(
        self,
        track: Track,
        source_track_id: int,
    ) -> None:
        # Set freeze on one track because source_track_id disappeared.
        track.set_cooldown(
            source_track_id=int(source_track_id),
            frames=self.cfg.overlap_app_freeze_after,
        )

    def freeze_track_near_unmatched(
        self,
        absent_ids: set[int],
        tracks: List[Track],
    ) -> set[int]:
        """
        If source track M disappeared and another track had M in overlap_group_ids,
        freeze that other track with source M.

        Also freeze M itself if it still exists.
        """
        tracks_by_id = {
            int(track.track_id): track
            for track in tracks
        }

        freshly_frozen_sources: set[int] = set()

        for absent_id in {int(x) for x in absent_ids}:
            affected_ids = {absent_id}

            for track in tracks:
                if absent_id in getattr(track, "overlap_group_ids", set()):
                    affected_ids.add(int(track.track_id))

            for affected_id in affected_ids:
                track = tracks_by_id.get(int(affected_id))

                if track is None:
                    continue

                self.set_cooldown(
                    track=track,
                    source_track_id=absent_id,
                )

            if affected_ids:
                freshly_frozen_sources.add(absent_id)

        return freshly_frozen_sources

    def which_defroze(
        self,
        matched_track_ids: set[int],
        tracks: List[Track],
    ) -> None:
        """
        If source track M matched again, clear source M from every track.

        Track N matching does not clear freeze caused by M.
        Only M matching clears freeze source M.
        """
        for source_track_id in {int(x) for x in matched_track_ids}:
            for track in tracks:
                track.clear_freeze_source(source_track_id)

    def decrease_freeze(
        self,
        tracks: List[Track],
        exclude_sources: Optional[set[int]] = None,
    ) -> None:
        # Decrease all active freezes by one frame.
        # Exclude sources that were created on this frame.
        exclude_sources = {int(x) for x in (exclude_sources or set())}

        for track in tracks:
            track.decrease_freeze(exclude_sources=exclude_sources)

    def update(
        self,
        detections: List[Detection],
        reset_mode: bool,
        g: float = 1.0,
    ) -> Dict[str, Any]:
        self._frame_idx += 1

        # Hard reset only on reset edge.
        # This avoids clearing tracks on every frame while reset_mode stays True.
        if reset_mode and not self._was_reset_mode:
            self.reset()

        # 1. Predict all active tracks.
        for trk in self.tracks:
            trk.predict()

        # Snapshot: row index -> track_id.
        idx2tid = {
            i: t.track_id
            for i, t in enumerate(self.tracks)
        }

        row_track_ids = [
            idx2tid[i]
            for i in range(len(self.tracks))
        ]

        # 2. Match tracks to detections.
        # Freeze/cooldown does NOT affect matching.
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

        # det_idx -> Track
        # Needed by adaptive overlap logic.
        det_idx_to_track = self._build_det_idx_to_track(matches_idx)

        matched_track_ids = set(current_matches.keys())

        # 3. If a frozen source returned, defreeze that source globally.
        self.which_defroze(
            matched_track_ids=matched_track_ids,
            tracks=self.tracks,
        )

        # 4. Detect tracks that disappeared compared to previous frame.
        absent_ids = self.compare_matches(
            prev_match=self.prev_matches,
            current_match=current_matches,
        )

        # 5. Freeze tracks that were near disappeared tracks on previous frame.
        freshly_frozen_sources = self.freeze_track_near_unmatched(
            absent_ids=absent_ids,
            tracks=self.tracks,
        )

        # 6. Update matched existing tracks.
        id_pairs: List[Tuple[int, int]] = []
        skipped_updates: List[Dict[str, Any]] = []

        max_update_cost = float(getattr(self.cfg.match, "max_update_cost", 1.2))
        max_update_motion = float(self.cfg.match.max_update_motion)
        max_update_pose = float(self.cfg.match.max_update_pose)
        max_update_app = float(self.cfg.match.max_update_app)

        track_update_debug: List[Dict[str, Any]] = []

        # C is row-relative matcher cost.
        # For update quality we use absolute raw weighted update_cost from matcher.py.
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

            # Standard matcher cost: row-relative cost.
            row_cost = float(C[i, j])

            # Absolute weighted raw cost.
            # This is used for Track update / skip-update decision.
            update_cost = float(update_cost_matrix[i, j])

            det.meta["match_cost"] = update_cost
            det.meta["match_row_cost"] = row_cost
            det.meta["match_update_cost"] = update_cost
            det.meta["match_d_motion"] = d_motion
            det.meta["match_d_pose"] = d_pose
            det.meta["match_d_app"] = d_app
            det.meta["match_update_threshold"] = max_update_cost
            det.meta["matched_track_id_before_update"] = int(trk.track_id)

            # Match still exists for visualization / outputs.
            id_pairs.append((trk.track_id, j_det))

            update_motion = d_motion <= max_update_motion
            update_pose = d_pose <= max_update_pose and self._has_base_keypoints(det)
            has_overlap = self._prepare_overlap_update_meta(
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
            # Decide appearance-memory update mode separately from matching.
            # Matching already happened; this only protects app_emb_ema.
            update_app_mode = str(app_decision["mode"])
            update_app = bool(app_decision["update_app"])

            disabled_reasons = []

            if not update_motion:
                disabled_reasons.append("motion_update_disabled")
            if not update_pose:
                disabled_reasons.append("pose_update_disabled")
            if not update_app:
                disabled_reasons.append("app_update_disabled")

            det.meta["track_update_skip_reason"] = (
                ",".join(disabled_reasons)
                if disabled_reasons
                else None
            )
            det.meta["track_update_disabled_reasons"] = disabled_reasons

            trk.update(
                det,
                ema_alpha=self.cfg.match.emb_ema_alpha,
                update_motion=update_motion,
                update_pose=update_pose,
                update_app=update_app,
                has_overlap=has_overlap,
            )
            # Finalize appearance stale counter after Track.update().
            # Strict updates are confirmed through det.meta["track_app_update_allowed"].
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
            det.meta.update({
                "app_update_mode": update_app_mode,
                "app_strict_update": app_decision["strict"], "app_recovery_candidate": app_decision["recovery_candidate"],
                "app_buffer_upper_eff": app_decision["buffer_upper_eff"], "app_buffer_base_upper": app_decision["buffer_base_upper"],
                "app_buffer_hard_upper": app_decision["buffer_hard_upper"], "app_stale_frames_before": app_decision["stale_before"],
                "app_stale_frames_after": app_decision["stale_after"], "app_buffer_size_before": app_decision["buffer_size_before"],
                "app_buffer_size_after": app_decision["buffer_size_after"], "app_buffer_min_size": int(getattr(self.cfg.match, "app_buffer_min_size", 3)),
                "app_buffer_clear_reason": app_decision["clear_reason"],
                "app_buffer_reject_reason": app_decision["reject_reason"], "app_coverage": app_decision["coverage"],
                "app_buffer_motion_ok": app_decision["motion_ok"], "app_buffer_pose_ok": app_decision["pose_ok"], "app_buffer_coverage_ok": app_decision["coverage_ok"],
                "app_buffer_overlap_ok": app_decision["overlap_ok"], "app_buffer_freeze_ok": app_decision["freeze_ok"],
                "app_recovery_batch_update_applied": app_decision["recovery_batch_applied"],
            })

            rec = {
                "track_idx": i,
                "track_id": int(trk.track_id),
                "det_idx": j,
                "hits_before_update": det.meta.get("track_hits_before_update"),
                "hits_after_update": int(trk.hits),
                "sub_confirmed_before_update": det.meta.get("track_sub_confirmed_before_update"),
                "sub_confirmed_after_update": bool(trk.sub_confirmed),
                "confirmed_before_update": det.meta.get("track_confirmed_before_update"),
                "confirmed_after_update": bool(trk.confirmed),
                "d_motion": d_motion,
                "d_pose": d_pose,
                "d_app": d_app,
                "max_update_motion": max_update_motion,
                "max_update_pose": max_update_pose,
                "max_update_app": max_update_app,
                "update_motion": bool(det.meta.get("track_update_motion_allowed", update_motion)),
                "update_pose": bool(det.meta.get("track_update_pose_allowed", update_pose)),
                "update_app": bool(det.meta.get("track_update_app_requested", update_app)),
                "track_match_had_overlap": bool(det.meta.get("track_match_had_overlap", False)),
                "birth_overlap_bypass": bool(det.meta.get("birth_overlap_bypass", False)),
                "row_cost": row_cost,
                "update_cost": update_cost,
                "max_update_cost": max_update_cost,
                "track_update_skipped": bool(det.meta.get("track_update_skipped", False)),
                "track_update_fully_skipped": bool(det.meta.get("track_update_fully_skipped", False)),
                "track_update_partially_skipped": bool(det.meta.get("track_update_partially_skipped", False)),
                "track_update_skip_reason": det.meta.get("track_update_skip_reason"),
                "track_app_update_allowed": bool(det.meta.get("track_app_update_allowed", False)),
                "track_app_update_block_reason": det.meta.get("track_app_update_block_reason"),
                "app_update_mode": det.meta.get("app_update_mode"),
                "app_buffer_upper_eff": det.meta.get("app_buffer_upper_eff"),
                "app_stale_frames_before": det.meta.get("app_stale_frames_before"),
                "app_stale_frames_after": det.meta.get("app_stale_frames_after"),
                "app_buffer_size_before": det.meta.get("app_buffer_size_before"),
                "app_buffer_size_after": det.meta.get("app_buffer_size_after"),
                "app_buffer_clear_reason": det.meta.get("app_buffer_clear_reason"),
                "app_buffer_reject_reason": det.meta.get("app_buffer_reject_reason"),
                "app_recovery_batch_update_applied": det.meta.get("app_recovery_batch_update_applied"),
                "max_overlap_iou": det.meta.get("max_overlap_iou"),
                "max_overlap_det_idx": det.meta.get("max_overlap_det_idx"),
                "min_center_dist_norm": det.meta.get("min_center_dist_norm"),
                "center_dist_norm_det_idx": det.meta.get("center_dist_norm_det_idx"),
                "active_overlap_threshold": det.meta.get("active_overlap_threshold"),
                "adaptive_overlap_zone": det.meta.get("adaptive_overlap_zone"),
                "adaptive_overlap_enabled": det.meta.get("adaptive_overlap_enabled"),
                "adaptive_overlap_reason": det.meta.get("adaptive_overlap_reason"),
                "current_track_stable_for_overlap": det.meta.get("current_track_stable_for_overlap"),
                "overlap_has_stable_track": det.meta.get("overlap_has_stable_track"),
                "track_has_risky_overlap": det.meta.get("track_has_risky_overlap"),
                "risky_overlap_count": det.meta.get("risky_overlap_count"),
                "risky_overlap_det_indices": det.meta.get("risky_overlap_det_indices"),
                "max_risky_overlap_iou": det.meta.get("max_risky_overlap_iou"),
                "raw_max_overlap_iou": det.meta.get("raw_max_overlap_iou"),
            }

            track_update_debug.append(rec)

            if rec["track_update_skipped"]:
                skipped_updates.append(rec)

        log.meta["used_config"] = self._used_config_debug()
        log.meta["track_update_debug"] = track_update_debug

        track_update_lines = format_track_update_debug_lines(track_update_debug)

        if track_update_lines and hasattr(log, "buffer") and isinstance(log.buffer, list):
            log.buffer.extend(["", *track_update_lines])

        # 7. Birth manager: unmatched detections become pending candidates first.
        birth_result = self.birth_manager.update(
            unmatched_det_indices=um_det_idx,
            detections=detections,
            existing_tracks=self.tracks,
            frame_idx=self._frame_idx,
            g=g,
        )

        log.meta["birth_debug"] = birth_result.debug_info
        append_birth_debug(birth_result.debug_info)

        if hasattr(log, "buffer") and isinstance(log.buffer, list):
            log.buffer.extend(["", *format_birth_debug_lines(birth_result.debug_info)])

        # 8. Spawn new tracks only for confirmed births.
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

        # 9. Build all current matches, including newly spawned tracks.
        all_current_matches: Dict[int, int] = {
            **current_matches,
            **new_matches,
        }

        # 10. Convert current detection-overlap groups into track-id groups.
        track_groups = self.dets_to_track(
            matches=all_current_matches,
            detections=detections,
        )

        # 11. Store overlap_group_ids inside each Track for the next frame.
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

        # Tracks that did not appear in current matches should not keep stale groups.
        for track in self.tracks:
            if int(track.track_id) not in track_groups:
                track.overlap_group_ids = set()

        # 12. Decrease active freeze counters.
        # Newly created freezes are excluded on this same frame.
        self.decrease_freeze(
            tracks=self.tracks,
            exclude_sources=freshly_frozen_sources,
        )

        freeze_debug = [
            {
                "track_idx": int(idx),
                "track_id": int(track.track_id),
                "freeze_active": bool(track.is_frozen()),
                "freeze_frames_left": int(getattr(track, "freeze_frames_left", 0)),
                "freeze_sources": {
                    int(source_id): int(frames_left)
                    for source_id, frames_left in getattr(track, "freeze_sources", {}).items()
                },
                "overlap_group_ids": sorted(
                    int(x)
                    for x in getattr(track, "overlap_group_ids", set())
                ),
            }
            for idx, track in enumerate(self.tracks)
        ]

        log.meta["freeze_debug"] = freeze_debug

        freeze_debug_lines = format_freeze_debug_lines(freeze_debug)

        if freeze_debug_lines and hasattr(log, "buffer") and isinstance(log.buffer, list):
            log.buffer.extend(["", *freeze_debug_lines])

        # 13. Save matches for next frame.
        self.prev_matches = all_current_matches

        # 14. Remove dead tracks.
        dead_tracks = self._remove_dead()
        log.meta["removed_dead_tracks"] = dead_tracks

        removed_lines = format_removed_tracks_lines(
            pruned=pruned_unconfirmed_tracks,
            dead=dead_tracks,
        )

        if removed_lines and hasattr(log, "buffer") and isinstance(log.buffer, list):
            log.buffer.extend(["", *removed_lines])

        unmatched_track_ids = sorted(int(x) for x in absent_ids)

        active_tracks_summary = [
            {
                "track_id": t.track_id,
                "sub_confirmed": t.sub_confirmed,
                "hits": t.hits,
                "confirmed": t.confirmed,
                "age": t.age,
                "time_since_update": t.time_since_update,
                "state": t.state.tolist(),
                "pos": t.pos(),
                "overlap_group_ids": sorted(
                    int(x)
                    for x in getattr(t, "overlap_group_ids", set())
                ),
                "freeze_sources": {
                    int(source_id): int(frames_left)
                    for source_id, frames_left in getattr(t, "freeze_sources", {}).items()
                },
                "freeze_frames_left": int(getattr(t, "freeze_frames_left", 0)),
                "app_emb_history_len": int(len(getattr(t, "app_emb_history", []) or [])),
            }
            for t in self.tracks
            if not t.is_dead(self.cfg.max_age, self.cfg.max_confirmed_age)
        ]

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
