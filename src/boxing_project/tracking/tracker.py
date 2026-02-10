from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from boxing_project.kalman_filter.kalman import KalmanTracker

from . import DEFAULT_TRACKING_CONFIG_PATH
from .matcher import MatchConfig, match_tracks_and_detections
from .track import Detection, Track


@dataclass
class TrackerConfig:
    dt: float
    process_var: float
    measure_var: float
    p0: float
    max_age: int
    min_hits: int
    match: MatchConfig
    reset_g_threshold: float
    debug: bool
    save_log: bool


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
            self.config_path: Optional[Path] = Path(config_path) if config_path is not None else DEFAULT_TRACKING_CONFIG_PATH
        else:
            self.cfg = copy.deepcopy(cfg)
            self._raw_config = None
            self.config_path = Path(config_path) if config_path is not None else None

        self.tracks: List[Track] = []
        self._next_id: int = 1
        raw = self.get_config_dict() or {}
        self.debug: bool = bool(raw.get("tracking", {}).get("debug", False))

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
        trk = Track(track_id=self._next_id, kf=kf, min_hits=self.cfg.min_hits)
        self._next_id += 1
        trk.update(det, ema_alpha=self.cfg.match.emb_ema_alpha, update_app=True)
        return trk

    def _remove_dead(self):
        self.tracks = [t for t in self.tracks if not t.is_dead(self.cfg.max_age)]

    def update(self, detections: List[Detection], reset_mode: bool, g: float = 1.0) -> Dict[str, Any]:
        for trk in self.tracks:
            trk.predict()

        idx2tid = {i: t.track_id for i, t in enumerate(self.tracks)}
        row_track_ids = [idx2tid[i] for i in range(len(self.tracks))]

        matches_idx, um_tr_idx, um_det_idx, C, log = match_tracks_and_detections(
            tracks=self.tracks,
            detections=detections,
            cfg=self.cfg.match,
            debug=self.debug,
            g=g,
            reset_mode=reset_mode,
        )

        id_pairs: List[Tuple[int, int]] = []
        for i_track, j_det in matches_idx:
            trk = self.tracks[i_track]
            det = detections[j_det]

            if reset_mode:
                trk.kf.reset(np.asarray(det.center, dtype=float), p0=float(self.cfg.p0))

            trk.update(det, ema_alpha=self.cfg.match.emb_ema_alpha, update_app=True)
            id_pairs.append((trk.track_id, j_det))

        for j in um_det_idx:
            if not reset_mode:
                self.tracks.append(self._new_track(detections[j]))

        self._remove_dead()

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
            "frame_log": log,
            "row_track_ids": row_track_ids,
        }

    def get_active_tracks(self, confirmed_only: bool = True) -> List[Track]:
        if confirmed_only:
            return [t for t in self.tracks if t.confirmed and not t.is_dead(self.cfg.max_age)]
        return [t for t in self.tracks if not t.is_dead(self.cfg.max_age)]
