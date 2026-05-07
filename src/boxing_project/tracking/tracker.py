from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np

from boxing_project.kalman_filter.kalman import KalmanTracker
from .track import Track, Detection
from .matcher import MatchConfig, match_tracks_and_detections
from .birth_manager import BirthConfig, BirthManager
from .tracking_debug import (
    append_birth_debug,
    format_birth_debug_lines,
    format_freeze_debug_lines,
    format_track_update_debug_lines,
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
    match: MatchConfig
    min_kp_conf: float
    reset_g_threshold: float
    debug: bool
    save_log: bool

    overlap_log_threshold: float = 0.10
    overlap_skip_threshold: float = 0.40
    overlap_app_freeze_after: int = 5

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
      5. which_defroze(): clear freeze sources for source tracks that returned.
      6. compare_matches(): find track ids that disappeared since prev_matches.
      7. freeze_track_near_unmatched(): start cooldown for tracks near disappeared sources.
      8. Update matched tracks.
      9. Spawn new tracks for unmatched detections.
      10. dets_to_track(): convert current detection-overlap groups into track-id groups.
      11. Write overlap_group_ids into Track objects for the next frame.
      12. decrease_freeze(): decrement active cooldown counters.
      13. Save prev_matches for the next frame.
      14. Remove dead tracks.

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

    def _new_track(self, det: Detection) -> Track:
        kf = KalmanTracker(
            x0=[det.center[0], det.center[1], 0.0, 0.0],
            dt=self.cfg.dt,
            process_var=self.cfg.process_var,
            measure_var=self.cfg.measure_var,
            p0=self.cfg.p0,
        )
        trk = Track(track_id=self._next_id, kf=kf, min_hits=self.cfg.min_hits, epoch_id=self._epoch_id)
        self._next_id += 1
        self._segment_tracks[trk.track_id] = trk
        self._epoch_tracks.setdefault(self._epoch_id, {})[trk.track_id] = trk
        trk.update(
            det,
            ema_alpha=self.cfg.match.emb_ema_alpha,
            update_app=self._has_base_keypoints(det),
            overlap_skip_threshold=self.cfg.overlap_skip_threshold,
        )
        return trk

    def _remove_dead(self):
        self.tracks = [t for t in self.tracks if not t.is_dead(self.cfg.max_age, self.cfg.max_confirmed_age)]

    def _has_base_keypoints(self, det: Detection, min_core_kps: Optional[int] = None) -> bool:
        """
        Check whether a detection contains enough valid keypoints from the configured core set.

        Modes:
          - Strict (min_core_kps is None): require ALL core keypoints to be valid.
          - Relaxed (min_core_kps is int): require at least `min_core_kps` valid core keypoints.

        A keypoint is considered valid if its (x, y) coordinates are finite.

        Returns:
            True  -> requirement satisfied
            False -> not satisfied
        """

        core = self.cfg.match.pose_core
        if not core:
            # No core configured => accept by default
            return True

        if det.keypoints is None:
            # No keypoints => cannot satisfy core requirement
            return False

        kps = np.asarray(det.keypoints, dtype=float)
        if kps.ndim != 2 or kps.shape[1] < 2:
            # Expect shape (K, >=2)
            return False

        # Keep only core indices that exist in this detection's keypoint array
        core = np.asarray(core, dtype=int).ravel()
        core = core[(0 <= core) & (core < kps.shape[0])]
        if core.size == 0:
            return False

        # Count valid core keypoints (finite x,y)
        valid_count = int(np.isfinite(kps[core, :2]).all(axis=1).sum())

        # Default behavior: strict mode (require all core keypoints)
        required = int(core.size) if min_core_kps is None else int(min_core_kps)

        return valid_count >= required

    def compare_matches(
        self,
        prev_match: Dict[int, int],
        current_match: Dict[int, int],
    ) -> set[int]:
        # Return track_ids that were matched on the previous frame
        # but are not matched on the current frame.
        return set(prev_match.keys()) - set(current_match.keys())

    def dets_to_track(
        self,
        matches: Dict[int, int],
        detections: List[Detection],
    ) -> Dict[int, set[int]]:
        # Convert detection overlap groups into track_id overlap groups.
        #
        # det.meta["overlap_relations"] stores det_idx.
        # For cooldown logic we need track_id.
        #
        # If an overlapped detection is not matched to any track, it is skipped.
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
            overlap_det_indices = Track.overlap_group(
                det,
                overlap_skip_threshold=self.cfg.overlap_skip_threshold,
            )

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
        # If source track M disappeared and another track had M in overlap_group_ids,
        # freeze that other track with source M.
        #
        # Also freeze M itself if it still exists.
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
        # If source track M matched again, clear source M from every track.
        #
        # Important:
        # Track N matching does not clear freeze caused by M.
        # Only M matching clears freeze source M.
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

    def update(self, detections: List[Detection], reset_mode: bool, g: float = 1.0) -> Dict[str, Any]:
        self._frame_idx += 1
        # Hard reset only on reset edge (False -> True).
        # This avoids clearing tracks on every frame while reset_mode stays True.
        if reset_mode and not self._was_reset_mode:
            self.reset()

        # 1) Predict step
        for trk in self.tracks:
            trk.predict()

        # snapshot: row index -> track_id
        idx2tid = {i: t.track_id for i, t in enumerate(self.tracks)}
        row_track_ids = [idx2tid[i] for i in range(len(self.tracks))]

        # 2) Match
        # Freeze/cooldown does NOT affect matching.
        matches_idx, um_tr_idx, um_det_idx, C, log = match_tracks_and_detections(
            tracks=self.tracks,
            detections=detections,
            cfg=self.cfg.match,
            debug=self.debug,
            g=g,
            reset_mode=reset_mode,
        )

        current_matches: Dict[int, int] = {
            int(self.tracks[i_track].track_id): int(j_det)
            for i_track, j_det in matches_idx
        }

        matched_track_ids = set(current_matches.keys())

        # 3) If a frozen source returned, defreeze that source globally.
        self.which_defroze(
            matched_track_ids=matched_track_ids,
            tracks=self.tracks,
        )

        # 4) Detect tracks that disappeared compared to previous frame.
        absent_ids = self.compare_matches(
            prev_match=self.prev_matches,
            current_match=current_matches,
        )

        # 5) Freeze tracks that were near disappeared tracks on previous frame.
        freshly_frozen_sources = self.freeze_track_near_unmatched(
            absent_ids=absent_ids,
            tracks=self.tracks,
        )

        # 6) Update matched existing tracks.
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
            # This can be 0 even for non-perfect match.
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
            update_pose = d_pose <= max_update_pose
            update_app = d_app <= max_update_app
            disabled_reasons = []
            if not update_motion:
                disabled_reasons.append("motion_update_disabled")
            if not update_pose:
                disabled_reasons.append("pose_update_disabled")
            if not update_app:
                disabled_reasons.append("app_update_disabled")

            det.meta["track_update_skip_reason"] = ",".join(disabled_reasons) if disabled_reasons else None
            det.meta["track_update_disabled_reasons"] = disabled_reasons

            trk.update(
                det,
                ema_alpha=self.cfg.match.emb_ema_alpha,
                update_motion=update_motion,
                update_pose=update_pose,
                update_app=update_app,
                overlap_skip_threshold=self.cfg.overlap_skip_threshold,
            )

            rec = {
                "track_idx": i,
                "track_id": int(trk.track_id),
                "det_idx": j,
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
                "row_cost": row_cost,
                "update_cost": update_cost,
                "max_update_cost": max_update_cost,
                "track_update_skipped": bool(det.meta.get("track_update_skipped", False)),
                "track_update_fully_skipped": bool(det.meta.get("track_update_fully_skipped", False)),
                "track_update_partially_skipped": bool(det.meta.get("track_update_partially_skipped", False)),
                "track_update_skip_reason": det.meta.get("track_update_skip_reason"),
                "track_app_update_allowed": bool(det.meta.get("track_app_update_allowed", False)),
                "track_app_update_block_reason": det.meta.get("track_app_update_block_reason"),
            }
            track_update_debug.append(rec)
            if rec["track_update_skipped"]:
                skipped_updates.append(rec)

        log.meta["track_update_debug"] = track_update_debug
        track_update_lines = format_track_update_debug_lines(track_update_debug)
        if track_update_lines and hasattr(log, "buffer") and isinstance(log.buffer, list):
            log.buffer.extend(["", *track_update_lines])

        # 7) Birth manager: unmatched detections become pending candidates first.
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

        # 8) Spawn new tracks only for confirmed births.
        new_matches: Dict[int, int] = {}
        for j in birth_result.confirmed_birth_det_indices:
            trk = self._new_track(detections[j])
            self.tracks.append(trk)
            new_matches[int(trk.track_id)] = int(j)

        # 9) Build all current matches, including newly spawned tracks.
        all_current_matches: Dict[int, int] = {
            **current_matches,
            **new_matches,
        }

        # 10) Convert current detection overlap groups to track_id overlap groups.
        track_groups = self.dets_to_track(
            matches=all_current_matches,
            detections=detections,
        )

        # 11) Store overlap_group_ids inside each Track for the next frame.
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

        # 12) Decrease active freeze counters.
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
                    int(x) for x in getattr(track, "overlap_group_ids", set())
                ),
            }
            for idx, track in enumerate(self.tracks)
        ]
        log.meta["freeze_debug"] = freeze_debug
        freeze_debug_lines = format_freeze_debug_lines(freeze_debug)
        if freeze_debug_lines and hasattr(log, "buffer") and isinstance(log.buffer, list):
            log.buffer.extend(["", *freeze_debug_lines])

        # 13) Save matches for the next frame.
        self.prev_matches = all_current_matches

        # 14) Remove dead
        self._remove_dead()

        unmatched_track_ids = sorted(int(x) for x in absent_ids)

        active_tracks_summary = [
            {
                "track_id": t.track_id,
                "confirmed": t.confirmed,
                "age": t.age,
                "hits": t.hits,
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
            return [t for t in self.tracks if t.confirmed and not t.is_dead(self.cfg.max_age, self.cfg.max_confirmed_age)]
        return [t for t in self.tracks if not t.is_dead(self.cfg.max_age, self.cfg.max_confirmed_age)]

    def get_segment_tracks(self) -> List[Track]:
        return list(self._segment_tracks.values())

    def reset(self):
        self.tracks.clear()
        self._segment_tracks.clear()
        self._next_id = 1

        self._epoch_id += 1
        self._epoch_tracks[self._epoch_id] = {}
        self.prev_matches.clear()
        self.birth_manager.pending.clear()

    def get_epoch_tracks(self) -> Dict[int, Dict[int, Track]]:
        return {epoch_id: dict(tracks_by_id) for epoch_id, tracks_by_id in self._epoch_tracks.items()}
