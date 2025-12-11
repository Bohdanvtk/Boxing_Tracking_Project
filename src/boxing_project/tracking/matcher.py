from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any, Union
import numpy as np
from scipy.optimize import linear_sum_assignment

from .track import Track, Detection
from . import DEFAULT_TRACKING_CONFIG_PATH

from .tracking_debug import (
    DebugLog,
    create_matcher_log,
    make_pair_base,
    fill_pair_gated_out,
    fill_pair_ok,
    print_gating_result,
    print_pair_result,
    set_pose_no_keypoints,
    set_pose_no_good_points,
    fill_pose_full_debug,
)



@dataclass
class MatchConfig:
    alpha: float
    chi2_gating: float
    large_cost: float
    pose_scale_eps: float
    keypoint_weights: Optional[np.ndarray]
    min_kp_conf: float


def _normalize_pose(kp: np.ndarray) -> Tuple[np.ndarray, float]:
    vis = np.isfinite(kp).all(axis=1)
    if not np.any(vis):
        return kp * 0.0, 1.0
    pts = kp[vis]
    center = pts.mean(axis=0, keepdims=True)
    centered = kp - center
    d = np.sqrt(((pts - center) ** 2).sum(axis=1) + 1e-12)
    scale = float(np.sqrt((d ** 2).mean()) + 1e-12)
    return centered / max(scale, 1e-12), scale


def _pose_distance(
    track: Track,
    det: Detection,
    cfg: MatchConfig,
    log: Optional[DebugLog] = None,
    pair_tag: str = ""
) -> Tuple[float, Dict[str, Any]]:
    """
    calculates d_pose, and return a dict for logs.
     (dist, pose_dict).
    """
    pose_dict: Dict[str, Any] = {}

    if track.last_keypoints is None or det.keypoints is None:
        set_pose_no_keypoints(pose_dict, log, pair_tag)
        return 0.0, pose_dict

    kpt_t = np.asarray(track.last_keypoints, dtype=float)
    kpt_d = np.asarray(det.keypoints, dtype=float)
    n_k = kpt_t.shape[0]

    conf_t = np.asarray(track.last_kp_conf, dtype=float).reshape(-1) if track.last_kp_conf is not None else np.ones((n_k,), dtype=float)
    conf_d = np.asarray(det.kp_conf, dtype=float).reshape(-1) if det.kp_conf is not None else np.ones((n_k,), dtype=float)

    assert kpt_t.shape[0] == kpt_d.shape[0], "Track/Det must have same number of keypoints"

    good_t = np.isfinite(kpt_t).all(axis=1) & (conf_t >= cfg.min_kp_conf)
    good_d = np.isfinite(kpt_d).all(axis=1) & (conf_d >= cfg.min_kp_conf)
    good = (good_t & good_d)

    if not np.any(good):
        set_pose_no_good_points(pose_dict, log, pair_tag, good)
        return 0.0, pose_dict

    # normalization
    kpt_tn, _ = _normalize_pose(kpt_t)
    kpt_dn, _ = _normalize_pose(kpt_d)

    diff_used = kpt_tn[good] - kpt_dn[good]
    per_k_used = np.linalg.norm(diff_used, axis=1)

    if cfg.keypoint_weights is not None:
        w_all = np.asarray(cfg.keypoint_weights, dtype=float).reshape(-1)
        w_used = w_all[good]
    else:
        w_all = np.ones((n_k,), dtype=float)
        w_used = np.ones_like(per_k_used)

    w_used = w_used * 0.5 * (conf_t[good] + conf_d[good])
    D_pose = float((w_used * per_k_used).sum() / (w_used.sum() + 1e-12))

    fill_pose_full_debug(
        pose_dict=pose_dict,
        log=log,
        pair_tag=pair_tag,
        n_k=n_k,
        good_mask=good,
        kpt_tn=kpt_tn,
        kpt_dn=kpt_dn,
        diff_used=diff_used,
        per_k_used=per_k_used,
        w_used=w_used,
        D_pose=D_pose,
    )

    return D_pose, pose_dict


def _motion_cost_with_gating(
    track: Track,
    det: Detection,
    cfg: MatchConfig,
    log: Optional[DebugLog] = None,
    pair_tag: str = ""
) -> Tuple[float, bool, float]:
    d2 = track.kf.gating_distance(np.asarray(det.center, dtype=float))
    allowed = (d2 <= cfg.chi2_gating)
    d_motion = float(np.sqrt(max(d2, 0.0)))

    if log and log.enabled_print:
        log.section(f"[{pair_tag}] MOTION")
        check = "✓" if allowed else "✗"
        log._print(f"• d² = {d2:.6f}   |   χ²_gate = {cfg.chi2_gating:.6f}   |   allowed = {check}")
        log._print(f"• d_motion = √(d²) = {d_motion:.6f}")


    return d_motion, allowed, float(d2)


def build_cost_matrix(
    tracks: List[Track],
    detections: List[Detection],
    cfg: MatchConfig,
    show: bool = True,
    sink: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n_tracks, n_dets = len(tracks), len(detections)
    shape = (n_tracks, n_dets)

    if n_tracks == 0 or n_dets == 0:
        log = create_matcher_log(cfg, shape, show=show, sink=sink)
        return np.zeros(shape, dtype=float), log.store

    C = np.zeros(shape, dtype=float)
    log = create_matcher_log(cfg, shape, show=show, sink=sink)

    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            pair_tag = f"T{i}↔D{j}"
            pair_obj: Dict[str, Any] = make_pair_base(i, j)

            d_motion, ok, d2 = _motion_cost_with_gating(
                trk, det, cfg, log=log, pair_tag=pair_tag
            )
            pair_obj["motion"] = {
                "d2": d2,
                "d_motion": d_motion,
                "allowed": bool(ok),
            }

            # 1) gating prunes a pair
            if not ok:
                C[i, j] = cfg.large_cost
                fill_pair_gated_out(pair_obj, cfg, d_motion)
                log.add_pair(pair_obj)
                print_gating_result(log, pair_tag, d2, cfg)
                continue

            # 2) pair ok → count D_pose and final cost
            d_pose, pose_dict = _pose_distance(
                trk, det, cfg, log=log, pair_tag=pair_tag
            )

            cost = cfg.alpha * d_motion + (1.0 - cfg.alpha) * d_pose
            C[i, j] = cost

            fill_pair_ok(
                pair_obj=pair_obj,
                cfg=cfg,
                d_motion=d_motion,
                d_pose=d_pose,
                cost=cost,
                pose_dict=pose_dict,
            )
            log.add_pair(pair_obj)
            print_pair_result(log, pair_tag, cfg, d_motion, d_pose, cost)



    return C, log.store


def linear_assignment_with_unmatched(C: np.ndarray, large_cost: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if C.size == 0:
        return [], list(range(C.shape[0])), list(range(C.shape[1]))

    rows, cols = linear_sum_assignment(C)

    matched = []
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


def _load_match_config_from_yaml(
    config_path: Optional[Union[str, Path]] = None,
) -> MatchConfig:
    """Read ``MatchConfig`` from a YAML file via ``utils.config`` helpers."""

    from src.boxing_project.utils.config import load_tracking_config

    resolved = Path(config_path) if config_path is not None else DEFAULT_TRACKING_CONFIG_PATH
    # ``load_tracking_config`` already creates the dataclass instance; copy to avoid
    # sharing mutable state between call sites.
    _, match_cfg, _ = load_tracking_config(str(resolved))
    return copy.deepcopy(match_cfg)


def match_tracks_and_detections(
    tracks: List[Track],
    detections: List[Detection],
    cfg: Optional[MatchConfig] = None,
    debug: bool = True,
    sink: Optional[Callable[[str], None]] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int], np.ndarray, Dict[str, Any]]:
    """
    Now return dict to log so that I will have a log on a frame level.

    """
    if cfg is None:
        cfg = _load_match_config_from_yaml(config_path)

    C, log_matcher = build_cost_matrix(tracks, detections, cfg, show=debug, sink=sink)
    matches, um_tr, um_dt = linear_assignment_with_unmatched(C, cfg.large_cost)
    return matches, um_tr, um_dt, C, log_matcher
