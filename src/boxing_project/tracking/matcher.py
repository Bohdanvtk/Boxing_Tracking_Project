from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from . import DEFAULT_TRACKING_CONFIG_PATH
from .track import Detection, Track
from .tracking_debug import DebugLog


@dataclass
class MatchConfig:
    alpha: float
    chi2_gating: float
    large_cost: float
    greedy_threshold: float
    greedy_reset_threshold: float
    emb_ema_alpha: float = 0.9
    w_motion: float = 1.0
    w_app: float = 1.0
    save_log: bool = False


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _motion_cost_with_gating(track: Track, det: Detection, cfg: MatchConfig, gating: bool) -> Tuple[float, bool, float]:
    d2 = track.kf.gating_distance(np.asarray(det.center, dtype=float))
    allowed = d2 <= cfg.chi2_gating
    d_motion = float(np.sqrt(max(d2, 0.0)))
    if not gating:
        allowed = True
    return d_motion, allowed, float(d2)


def build_cost_matrix(
    tracks: List[Track],
    detections: List[Detection],
    reset_mode: bool,
    cfg: MatchConfig,
    show: bool = True,
    sink: Optional[Callable[[str], None]] = None,
    g: float = 1.0,
) -> Tuple[np.ndarray, DebugLog]:
    n_t = len(tracks)
    n_d = len(detections)
    C = np.zeros((n_t, n_d), dtype=np.float32)

    g = float(np.clip(g, 0.0, 1.0))
    log = DebugLog(enabled_print=show, sink=sink or print)
    log.meta = {
        "g": g,
        "tracks": tracks,
        "detections": detections,
        "track_ids": [t.track_id for t in tracks],
        "det_ids": list(range(len(detections))),
    }
    log.create_matrix(n_t, n_d)

    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            cell = log[i, j]

            if reset_mode:
                d2 = 0.0
                allowed = True
                d_motion_norm = 0.0
            else:
                _, allowed, d2 = _motion_cost_with_gating(trk, det, cfg, gating=True)
                d_motion_norm = float(np.clip(d2 / max(cfg.chi2_gating, 1e-12), 0.0, 1.0))

            cell.d_motion = float(d_motion_norm)
            cell.allowed = bool(allowed)
            cell.d2 = float(d2)

            if not allowed:
                C[i, j] = float(cfg.large_cost)
                cell.cost = float(cfg.large_cost)
                continue

            if trk.app_emb_ema is not None and isinstance(det.meta, dict) and ("e_app" in det.meta):
                sim_app = cosine_similarity(trk.app_emb_ema, det.meta["e_app"])
            else:
                sim_app = 0.0
            app_cost = (1.0 - sim_app) / 2.0

            if reset_mode:
                cost = cfg.w_app * app_cost
            else:
                cost = cfg.w_motion * (g * d_motion_norm) + cfg.w_app * app_cost

            cell.d_pose = 0.0
            cell.d_app = float(app_cost)
            cell.cost = float(cost)
            C[i, j] = float(cost)

    if show or cfg.save_log:
        log.show_matrix()

    return C, log


def linear_assignment_with_unmatched(
    C: np.ndarray,
    large_cost: float,
    reset_mode: bool,
    greedy_threshold: float = 2.8,
    greedy_reset_threshold: float = 1.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    thr = greedy_reset_threshold if reset_mode else greedy_threshold

    if C.size == 0:
        return [], list(range(C.shape[0])), list(range(C.shape[1]))

    n_tracks, n_dets = C.shape

    edges: List[Tuple[float, int, int]] = []
    for r in range(n_tracks):
        for c in range(n_dets):
            cost = float(C[r, c])
            if cost >= large_cost or cost > thr:
                continue
            edges.append((cost, r, c))

    edges.sort(key=lambda x: x[0])

    matched: List[Tuple[int, int]] = []
    used_tracks = set()
    used_dets = set()
    for cost, r, c in edges:
        if r in used_tracks or c in used_dets:
            continue
        matched.append((int(r), int(c)))
        used_tracks.add(r)
        used_dets.add(c)

    unmatched_tracks = [i for i in range(n_tracks) if i not in used_tracks]
    unmatched_dets = [j for j in range(n_dets) if j not in used_dets]
    return matched, unmatched_tracks, unmatched_dets


def _load_match_config_from_yaml(config_path: Optional[Union[str, Path]] = None) -> MatchConfig:
    from boxing_project.utils.config import load_tracking_config

    resolved = Path(config_path) if config_path is not None else DEFAULT_TRACKING_CONFIG_PATH
    _, match_cfg, _ = load_tracking_config(str(resolved))
    return copy.deepcopy(match_cfg)


def match_tracks_and_detections(
    tracks,
    detections,
    reset_mode: bool,
    cfg=None,
    debug: bool = True,
    sink=None,
    config_path=None,
    g: float = 1.0,
):
    if cfg is None:
        cfg = _load_match_config_from_yaml(config_path)

    C, log = build_cost_matrix(
        tracks=tracks,
        detections=detections,
        cfg=cfg,
        show=debug,
        sink=sink,
        g=g,
        reset_mode=reset_mode,
    )

    matches, um_tr, um_dt = linear_assignment_with_unmatched(
        C,
        large_cost=cfg.large_cost,
        greedy_threshold=cfg.greedy_threshold,
        greedy_reset_threshold=cfg.greedy_reset_threshold,
        reset_mode=reset_mode,
    )
    return matches, um_tr, um_dt, C, log
