from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from .track import Track, Detection
from . import DEFAULT_TRACKING_CONFIG_PATH
from .tracking_debug import DebugLog


@dataclass
class MatchConfig:
    """"Конфіг для побудови cost matrix (motion/pose/app + gating)." """
    alpha: float
    chi2_gating: float
    large_cost: float
    pose_scale_eps: float
    keypoint_weights: Optional[np.ndarray]
    min_kp_conf: float

    emb_ema_alpha: float = 0.9

    w_motion: float = float
    w_pose: float = float
    w_app: float = float

    save_log: bool = bool


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    vector similarity [-1:1]
        -1: not similar
         1: similar
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(np.dot(a, b) / (na * nb))




def _motion_cost_with_gating(track: Track, det: Detection, cfg: MatchConfig, gating: bool) -> Tuple[float, bool, float]:
    d2 = track.kf.gating_distance(np.asarray(det.center, dtype=float))
    allowed = (d2 <= cfg.chi2_gating)
    d_motion = float(np.sqrt(max(d2, 0.0)))

    if gating == False:
        allowed = True
    return d_motion, allowed, float(d2)


def build_cost_matrix(
    tracks: List[Track],
    detections: List[Detection],
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
        "tracks": tracks,  # <-- щоб _get_track_id(i) брав trk.track_id
        "detections": detections,  # <-- щоб _get_det_id(j) міг брати id якщо є
        "track_ids": [t.track_id for t in tracks],  # (опційно, але супер-надійно)
        "det_ids": list(range(len(detections)))  # щоб явно було Det#j == j
    }

    log.create_matrix(n_t, n_d)

    gating = True

    if g < 0.79:
        g = g ** 3 # <= головне: low g => bigger motion cost
        gating = False


    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            cell = log[i, j]

            d_motion, allowed, d2 = _motion_cost_with_gating(trk, det, cfg, gating)


            d_motion_norm = float(np.clip(d2 / max(cfg.chi2_gating, 1e-12), 0.0, 1.0))
            cell.d_motion = float(d_motion_norm)
            cell.allowed = bool(allowed)
            cell.d2 = float(d2)

            if not allowed:
                C[i, j] = float(cfg.large_cost)
                cell.cost = float(cfg.large_cost)
                continue

            # pose (embedding або fallback)
            if trk.pose_emb_ema is not None and isinstance(det.meta, dict) and ("e_pose" in det.meta):
                d_pose = cosine_distance(trk.pose_emb_ema, det.meta["e_pose"])
            else:
                d_pose = 0.0

            # appearance (embedding або 0)
            if trk.app_emb_ema is not None and isinstance(det.meta, dict) and ("e_app" in det.meta):
                d_app = cosine_distance(trk.app_emb_ema, det.meta["e_app"])
            else:
                d_app = 0.0

            d_pose_norm = (1.0 - d_pose) / 2.0
            d_app_norm = (1.0 - d_app) / 2.0

            cost = (
                cfg.w_motion * (g * d_motion_norm)
                + cfg.w_pose * d_pose_norm
                + cfg.w_app * d_app_norm
            )

            cell.d_pose = float(d_pose_norm)
            cell.d_app = float(d_app_norm)
            cell.cost = float(cost)

            C[i, j] = float(cost)

    if show or cfg.save_log:
        log.show_matrix()

    return C, log

'''
def linear_assignment_with_unmatched(C: np.ndarray, large_cost: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if C.size == 0:
        return [], list(range(C.shape[0])), list(range(C.shape[1]))

    rows, cols = linear_sum_assignment(C)

    matched: List[Tuple[int, int]] = []
    used_tracks = set()
    used_dets = set()

    for r, c in zip(rows, cols):
        if C[r, c] >= large_cost:
            continue
        matched.append((int(r), int(c)))
        used_tracks.add(int(r))
        used_dets.add(int(c))

    n_tracks, n_dets = C.shape
    unmatched_tracks = [i for i in range(n_tracks) if i not in used_tracks]
    unmatched_dets = [j for j in range(n_dets) if j not in used_dets]

    return matched, unmatched_tracks, unmatched_dets
'''

def linear_assignment_with_unmatched(
    C: np.ndarray,
    large_cost: float,
    greedy_threshold: float = 1.8,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Global greedy assignment:
    - build list of all (track, det, cost) where cost < greedy_threshold
      (and also skip large_cost if you still use it as "invalid match")
    - sort by cost ascending
    - take smallest edges while keeping one-to-one constraint
    - return matched pairs + unmatched tracks/dets

    This implements "smaller cost = higher priority" directly.
    """

    if C.size == 0:
        # Note: for empty cost matrix, shapes still matter
        return [], list(range(C.shape[0])), list(range(C.shape[1]))

    n_tracks, n_dets = C.shape

    # Collect all candidate edges
    edges: List[Tuple[float, int, int]] = []
    for r in range(n_tracks):
        for c in range(n_dets):
            cost = float(C[r, c])
            # Skip invalid / gated entries
            if cost >= large_cost:
                continue
            # Hard accept/reject threshold
            if cost > greedy_threshold:
                continue
            edges.append((cost, r, c))

    # Sort by increasing cost (best matches first)
    edges.sort(key=lambda x: x[0])

    matched: List[Tuple[int, int]] = []
    used_tracks = set()
    used_dets = set()

    # Greedy pick
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
    )
    matches, um_tr, um_dt = linear_assignment_with_unmatched(C, cfg.large_cost)
    return matches, um_tr, um_dt, C, log
