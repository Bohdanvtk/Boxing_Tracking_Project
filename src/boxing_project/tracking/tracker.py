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
from . import DEFAULT_TRACKING_CONFIG_PATH


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
    reset_g_threshold: float
    bad_kp_patience: int
    debug: bool
    save_log: bool


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
        trk.update(
            det,
            ema_alpha=self.cfg.match.emb_ema_alpha,
            update_app=self._has_base_keypoints(det),
        )
        return trk

    def _remove_dead(self):
        self.tracks = [t for t in self.tracks if not t.is_dead(self.cfg.max_age)]

    def _has_base_keypoints(self, det: Detection) -> bool:
        core = np.asarray(self.cfg.match.pose_core, dtype=int).reshape(-1) if self.cfg.match.pose_core is not None else None
        if core is None or core.size == 0:
            return True
        if det.keypoints is None:
            return False
        kps = np.asarray(det.keypoints, dtype=float)
        if kps.ndim != 2 or kps.shape[1] < 2:
            return False
        n_k = kps.shape[0]
        core = core[(core >= 0) & (core < n_k)]
        if core.size == 0:
            return False
        return bool(np.isfinite(kps[core, :2]).all(axis=1).all())



    def update(self, detections: List[Detection], reset_mode: bool, g: float = 1.0) -> Dict[str, Any]:
        # 1) predict
        for trk in self.tracks:
            trk.predict()

        # snapshot: row index -> track_id
        idx2tid = {i: t.track_id for i, t in enumerate(self.tracks)}
        row_track_ids = [idx2tid[i] for i in range(len(self.tracks))]

        # 2) match (returns DebugLog as last element)
        matches_idx, um_tr_idx, um_det_idx, C, log = match_tracks_and_detections(
            tracks=self.tracks,
            detections=detections,
            cfg=self.cfg.match,
            debug=self.debug,
            g=g,
            reset_mode=reset_mode,
        )


        # 3) update matched
        id_pairs: List[Tuple[int, int]] = []
        for i_track, j_det in matches_idx:
            trk = self.tracks[i_track]
            det = detections[j_det]

            if reset_mode:
                # Hard reset of motion state
                trk.kf.reset(np.asarray(det.center, dtype=float), p0=float(self.cfg.p0))
                trk.last_keypoints = None
                trk.last_kp_conf = None
                trk.post_reset_mode = True
                trk.bad_kp_streak = 0

            has_core = self._has_base_keypoints(det)

            if not trk.post_reset_mode:
                # Normal behavior before any reset
                trk.update(
                    det,
                    ema_alpha=self.cfg.match.emb_ema_alpha,
                    update_app=True,
                )
            else:
                # After reset, apply core-keypoint gating with patience
                if has_core:
                    trk.bad_kp_streak = 0
                    trk.update(
                        det,
                        ema_alpha=self.cfg.match.emb_ema_alpha,
                        update_app=True,
                    )
                else:
                    trk.bad_kp_streak += 1

                    if trk.bad_kp_streak <= int(self.cfg.bad_kp_patience):
                        # Skip state update but keep track active
                        trk.time_since_update = 0
                    else:
                        # Force update after exceeding patience
                        trk.update(
                            det,
                            ema_alpha=self.cfg.match.emb_ema_alpha,
                            update_app=True,
                        )

            id_pairs.append((trk.track_id, j_det))

        # 4) spawn new tracks
        for j in um_det_idx:

            if not reset_mode:
                self.tracks.append(self._new_track(detections[j]))

        # 5) remove dead
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

        # return consistent structure
        return {
            "matches": id_pairs,
            "unmatched_track_ids": unmatched_track_ids,
            "unmatched_det_indices": um_det_idx,
            "cost_matrix": C,
            "active_tracks": active_tracks_summary,
            "frame_log": log,           # DebugLog (matrix-based)
            "row_track_ids": row_track_ids,
        }

    def get_active_tracks(self, confirmed_only: bool = True) -> List[Track]:
        if confirmed_only:
            return [t for t in self.tracks if t.confirmed and not t.is_dead(self.cfg.max_age)]
        return [t for t in self.tracks if not t.is_dead(self.cfg.max_age)]

    def reset(self):
        self.tracks.clear()
        self._next_id = 1
