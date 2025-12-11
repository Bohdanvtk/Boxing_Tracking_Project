from __future__ import annotations
import copy
import numbers
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from collections import defaultdict
import numpy as np

from .kalman import KalmanTracker
from .track import Track, Detection
from .matcher import MatchConfig, match_tracks_and_detections
from . import DEFAULT_TRACKING_CONFIG_PATH


# ----------------------------- #
#          Tracker config       #
# ----------------------------- #

@dataclass
class TrackerConfig:
    dt: float
    process_var: float
    measure_var: float
    p0: float
    max_age: int
    min_hits: int
    match: MatchConfig
    min_kp_conf: float
    expect_body25: bool

# ----------------------------- #
#     Adapter OpenPose → Det    #
# ----------------------------- #

def openpose_people_to_detections(
    people: List[Dict[str, Any]],
    min_kp_conf: float = 0.05,
    expect_body25: bool = True
) -> List[Detection]:
    """
    Converts a list of people from OpenPose JSON into a list of Detection objects.
    Supports several formats:
      1) 'pose_keypoints_2d' — flattened list of length 75 (BODY_25: 25*(x,y,conf))
      2) 'keypoints'/'pose' as np.ndarray or list with shape (K, 3)
      3) 'pose_2d' with fields 'x', 'y', 'conf' (any form that can be reshaped to (K,3))

    Detection center = median of visible (x,y) with conf >= min_kp_conf (robust to outliers).
    """
    dets: List[Detection] = []

    for person in people:
        kps: Optional[np.ndarray] = None

        # option 1: standard OpenPose JSON
        if 'pose_keypoints_2d' in person and isinstance(person['pose_keypoints_2d'], (list, tuple)):
            arr = np.asarray(person['pose_keypoints_2d'], dtype=float).reshape(-1)
            if arr.size % 3 != 0:
                # incorrect size — skip this person
                continue
            K = arr.size // 3
            # if BODY_25 is expected, K should be 25; but allow others to avoid crashing
            kps = arr.reshape(K, 3)

        # option 2: already provided as (K,3) or (K,2)
        elif 'keypoints' in person:
            arr = np.asarray(person['keypoints'], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                if arr.shape[1] == 2:
                    # no confidence -> add conf=1
                    ones = np.ones((arr.shape[0], 1), dtype=float)
                    arr = np.concatenate([arr, ones], axis=1)
                kps = arr[:, :3]

        elif 'pose' in person:
            arr = np.asarray(person['pose'], dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                if arr.shape[1] == 2:
                    ones = np.ones((arr.shape[0], 1), dtype=float)
                    arr = np.concatenate([arr, ones], axis=1)
                kps = arr[:, :3]

        elif 'pose_2d' in person:
            p = person['pose_2d']
            xs = np.asarray(p.get('x', []), dtype=float).reshape(-1, 1)
            ys = np.asarray(p.get('y', []), dtype=float).reshape(-1, 1)
            cs = np.asarray(p.get('conf', []), dtype=float).reshape(-1, 1)
            if xs.shape == ys.shape == cs.shape and xs.size > 0:
                kps = np.concatenate([xs, ys, cs], axis=1)

        if kps is None:
            # no keypoints — skip
            continue

        # Filter by conf: hide poor ones as NaN (so they don't affect calculations)
        good = kps[:, 2] >= float(min_kp_conf)
        xy = kps[:, :2].copy()
        xy[~good] = np.nan

        # Center — median of visible points (robust against outliers)
        if np.all(~good):
            # if no reliable points — skip this person
            continue
        cx = np.nanmedian(xy[:, 0])
        cy = np.nanmedian(xy[:, 1])

        dets.append(
            Detection(
                center=(float(cx), float(cy)),
                keypoints=xy,      # (K, 2) with NaN where conf is low
                kp_conf=kps[:, 2], # (K,)
                meta={'raw': person}
            )
        )

    return dets


# ----------------------------- #
#            TRACKER            #
# ----------------------------- #

@lru_cache(maxsize=None)
def _cached_tracking_config(path: str):
    """Cache loader so multiple trackers reuse the parsed YAML."""

    from src.boxing_project.utils.config import load_tracking_config

    return load_tracking_config(path)


def _load_tracker_config_from_yaml(
    config_path: Optional[Union[str, Path]] = None,
) -> Tuple[TrackerConfig, Dict[str, Any]]:
    """Load ``TrackerConfig`` and raw dictionary from YAML file."""

    resolved = Path(config_path) if config_path is not None else DEFAULT_TRACKING_CONFIG_PATH
    tracker_cfg, match_cfg, raw_cfg = _cached_tracking_config(str(resolved))
    tracker_cfg_copy = copy.deepcopy(tracker_cfg)
    # Ensure nested MatchConfig is also unique per tracker instance.
    tracker_cfg_copy.match = copy.deepcopy(match_cfg)
    return tracker_cfg_copy, copy.deepcopy(raw_cfg)


def resolve_show_level(value: Any) -> int:
    """Convert configuration values to the logging level 0/1/2."""

    if isinstance(value, bool):
        return 2 if value else 0

    if value is None:
        return 1

    if isinstance(value, numbers.Integral):
        level = int(value)
    elif isinstance(value, str):
        value = value.strip()
        if value == "":
            return 1
        try:
            level = int(value)
        except ValueError as exc:
            raise ValueError(
                "tracking.show must be an integer 0, 1 or 2"
            ) from exc
    else:
        raise ValueError("tracking.show must be an integer 0, 1 or 2")

    if level not in (0, 1, 2):
        raise ValueError("tracking.show must be one of {0, 1, 2}")

    return level


class MultiObjectTracker:
    """
    Manages a list of tracks:
      predict → build C → Hungarian → update → track lifecycle management.

    If ``cfg`` is not provided, the settings are loaded from the YAML file through
    ``utils.config.load_tracking_config`` (``config_path`` or default
    ``DEFAULT_TRACKING_CONFIG_PATH``).
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
        self._next_id: int = 1

        raw_cfg = self.get_config_dict() or {}
        show_value = raw_cfg.get("tracking", {}).get("show", None)
        self.show_level: int = resolve_show_level(show_value)
        self.show_debug: bool = self.show_level >= 2

    # ---- utility methods ---- #

    def get_config_dict(self) -> Optional[Dict[str, Any]]:
        """Return a deep copy of the raw YAML configuration used for this tracker."""

        if self._raw_config is None:
            return None
        return copy.deepcopy(self._raw_config)

    def _new_track(self, det: Detection) -> Track:
        kf = KalmanTracker(
            x0=[det.center[0], det.center[1], 0.0, 0.0],
            dt=self.cfg.dt,
            process_var=self.cfg.process_var,
            measure_var=self.cfg.measure_var,
            p0=self.cfg.p0
        )
        trk = Track(
            track_id=self._next_id,
            kf=kf,
            min_hits=self.cfg.min_hits
        )
        self._next_id += 1
        # immediate first update — to align state with initial measurement
        trk.update(det)
        return trk

    def _remove_dead(self):
        self.tracks = [t for t in self.tracks if not t.is_dead(self.cfg.max_age)]

    # ---- per-frame API ---- #

    def update_with_openpose(
        self,
        openpose_people: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Full cycle for a single frame with raw people from OpenPose.

        Returns a dictionary:
          {
            'matches': List[Tuple[track_id, det_index]],
            'unmatched_track_ids': List[int],
            'unmatched_det_indices': List[int],
            'cost_matrix': np.ndarray,
            'active_tracks': List[Dict]  # brief state of tracks after update
          }
        """
        detections = openpose_people_to_detections(
            openpose_people,
            min_kp_conf=self.cfg.min_kp_conf,
            expect_body25=self.cfg.expect_body25
        )
        return self.update(detections)

    def update(
            self,
            detections: List[Detection]
    ) -> Dict[str, Any]:
        # 1) PREDICT
        for trk in self.tracks:
            trk.predict()

        # --- SNAPSHOT: index of track → track_id BEFORE matching ---
        idx2tid = {i: t.track_id for i, t in enumerate(self.tracks)}
        # >>> ADD THIS: list of track_ids in the same order as rows of cost_matrix
        row_track_ids = [idx2tid[i] for i in range(len(self.tracks))]

        # 2) MATCH
        matches_idx, um_tr_idx, um_det_idx, C, log_matcher = match_tracks_and_detections(
            tracks=self.tracks,
            detections=detections,
            cfg=self.cfg.match,
            debug=self.show_debug,
        )

        # 3) collect pair_logs_by_tid ...
        pair_logs_by_tid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for p in log_matcher.get("pairs", []):
            i = p.get("track_index")
            tid = idx2tid.get(i)
            if tid is not None:
                pair_logs_by_tid[tid].append(copy.deepcopy(p))

        # 4) UPDATE assigned tracks ...
        id_pairs: List[Tuple[int, int]] = []
        for i_track, j_det in matches_idx:
            trk = self.tracks[i_track]
            det = detections[j_det]
            trk.update(det)
            id_pairs.append((trk.track_id, j_det))

        # 5) create new tracks from unmatched detections ...
        for j in um_det_idx:
            new_trk = self._new_track(detections[j])
            self.tracks.append(new_trk)
            pair_logs_by_tid[new_trk.track_id].append({
                "track_index": None,
                "det_index": j,
                "motion": None,
                "pose": None,
                "final": {
                    "alpha": self.cfg.match.alpha,
                    "cost": 0.0,
                    "components": {"d_motion": 0.0, "d_pose": 0.0},
                    "reason": "new_track_from_unmatched_detection"
                }
            })

        # 6) remove dead tracks
        self._remove_dead()

        # 7) collect output
        unmatched_track_ids = [idx2tid[i] for i in um_tr_idx if i in idx2tid]

        active_tracks_summary = [
            {
                "track_id": t.track_id,
                "confirmed": t.confirmed,
                "age": t.age,
                "hits": t.hits,
                "time_since_update": t.time_since_update,
                "state": t.state.tolist(),
                "pos": t.pos(),
                "match_log": pair_logs_by_tid.get(t.track_id, [])
            }
            for t in self.tracks
            if not t.is_dead(self.cfg.max_age)
        ]

        return {
            "matches": id_pairs,
            "unmatched_track_ids": unmatched_track_ids,
            "unmatched_det_indices": um_det_idx,
            "cost_matrix": C,
            "active_tracks": active_tracks_summary,
            "frame_log": log_matcher,
            "row_track_ids": row_track_ids  # ← now defined above
        }

    # ---- utilities ---- #

    def get_active_tracks(
        self,
        confirmed_only: bool = True
    ) -> List[Track]:
        """Return a list of active tracks (by default only confirmed ones)."""
        if confirmed_only:
            return [t for t in self.tracks if t.confirmed and not t.is_dead(self.cfg.max_age)]
        return [t for t in self.tracks if not t.is_dead(self.cfg.max_age)]

    def reset(self):
        """Reset all tracks (new video)."""
        self.tracks.clear()
        self._next_id = 1
