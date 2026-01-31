from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Union, Dict, Any

import numpy as np

from .track import Track, Detection
from . import DEFAULT_TRACKING_CONFIG_PATH
from .tracking_debug import DebugLog
from boxing_project.tracking.normalization import normalize_pose_2d


@dataclass
class MatchConfig:
    """"Конфіг для побудови cost matrix (motion/pose/app + gating)." """
    alpha: float
    chi2_gating: float
    large_cost: float
    pose_scale_eps: float
    keypoint_weights: Optional[np.ndarray]
    min_kp_conf: float
    greedy_threshold: float
    greedy_reset_threshold: float
    emb_ema_alpha: float = 0.9

    w_motion: float = float
    w_pose: float = float
    w_app: float = float
    pose_core: Optional[List[int]] = None

    debug_pose_extended: bool = False
    debug_pose_print_table: bool = False

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
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pose_distance(
    track_keypoints: Optional[np.ndarray],
    track_kp_conf: Optional[np.ndarray],
    det_keypoints: Optional[np.ndarray],
    det_kp_conf: Optional[np.ndarray],
    cfg: MatchConfig,
    log: Optional[DebugLog] = None,
    pair_tag: str = "",
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    Pose similarity (cosine) computed only on cfg.pose_core joints.

    Returns cosine similarity in [-1..1].
    If nothing usable -> returns 0.0
    """

    if track_keypoints is None or det_keypoints is None:
        return 0.0, None

    kpt_t = np.asarray(track_keypoints, dtype=float)
    kpt_d = np.asarray(det_keypoints, dtype=float)

    if kpt_t.ndim != 2 or kpt_d.ndim != 2 or kpt_t.shape[1] < 2 or kpt_d.shape[1] < 2:
        return 0.0, None
    if kpt_t.shape[0] != kpt_d.shape[0]:
        return 0.0, None

    n_k = kpt_t.shape[0]

    conf_t = (
        np.asarray(track_kp_conf, dtype=float).reshape(-1)
        if track_kp_conf is not None else np.ones((n_k,), dtype=float)
    )
    conf_d = (
        np.asarray(det_kp_conf, dtype=float).reshape(-1)
        if det_kp_conf is not None else np.ones((n_k,), dtype=float)
    )
    if conf_t.shape[0] != n_k or conf_d.shape[0] != n_k:
        return 0.0, None

    trk_present = np.isfinite(kpt_t[:, :2]).all(axis=1) & (conf_t >= cfg.min_kp_conf)
    det_present = np.isfinite(kpt_d[:, :2]).all(axis=1) & (conf_d >= cfg.min_kp_conf)
    used_mask = trk_present & det_present

    dx = np.full((n_k,), np.nan, dtype=float)
    dy = np.full((n_k,), np.nan, dtype=float)
    norm = np.full((n_k,), np.nan, dtype=float)
    valid_idx = np.where(used_mask)[0]
    if valid_idx.size > 0:
        delta = kpt_d[valid_idx, :2] - kpt_t[valid_idx, :2]
        dx[valid_idx] = delta[:, 0]
        dy[valid_idx] = delta[:, 1]
        norm[valid_idx] = np.linalg.norm(delta, axis=1)

    used_idx = valid_idx.tolist()
    used_count = int(len(used_idx))
    d_pose_mean = float(np.nanmean(norm[used_mask])) if used_count > 0 else 0.0

    pose_dict: Optional[Dict[str, Any]] = None
    if cfg.debug_pose_extended:
        pose_dict = {
            "trk_xy": kpt_t[:, :2].tolist(),
            "det_xy": kpt_d[:, :2].tolist(),
            "trk_conf": conf_t.tolist(),
            "det_conf": conf_d.tolist(),
            "trk_present": trk_present.tolist(),
            "det_present": det_present.tolist(),
            "used_mask": used_mask.tolist(),
            "dx": dx.tolist(),
            "dy": dy.tolist(),
            "norm": norm.tolist(),
            "used_idx": used_idx,
            "used_count": used_count,
            "D_pose": d_pose_mean,
        }

        core = np.asarray(cfg.pose_core, dtype=int).reshape(-1) if cfg.pose_core is not None else None
        if core is not None and core.size > 0:
            core = core[(core >= 0) & (core < n_k)]
            if core.size > 0:
                core_present = trk_present[core] & det_present[core]
                core_present_idx = core[core_present].tolist()
                core_present_count = int(len(core_present_idx))
                core_total = int(core.size)
                core_ratio = core_present_count / core_total if core_total > 0 else None
                pose_dict.update(
                    {
                        "core_present_idx": core_present_idx,
                        "core_present_count": core_present_count,
                        "core_total": core_total,
                        "core_ratio": core_ratio,
                    }
                )

        if cfg.debug_pose_print_table and log is not None and log.enabled_print:
            header = f"[pose-debug] {pair_tag}" if pair_tag else "[pose-debug]"
            log.sink(header)
            log.sink(
                "k | trk_x trk_y | det_x det_y | conf_t conf_d | trk_ok det_ok used | dx dy | ||d||"
            )
            for k in range(n_k):
                log.sink(
                    f"{k:02d} | "
                    f"{kpt_t[k, 0]:.4f} {kpt_t[k, 1]:.4f} | "
                    f"{kpt_d[k, 0]:.4f} {kpt_d[k, 1]:.4f} | "
                    f"{conf_t[k]:.3f} {conf_d[k]:.3f} | "
                    f"{int(trk_present[k])} {int(det_present[k])} {int(used_mask[k])} | "
                    f"{dx[k]:.4f} {dy[k]:.4f} | "
                    f"{norm[k]:.4f}"
                )

    pose_similarity = 0.0
    core = np.asarray(cfg.pose_core, dtype=int).reshape(-1) if cfg.pose_core is not None else None
    if core is not None and core.size > 0:
        core = core[(core >= 0) & (core < n_k)]
        if core.size > 0:
            good_t = np.isfinite(kpt_t[core, :2]).all(axis=1) & (conf_t[core] >= cfg.min_kp_conf)
            good_d = np.isfinite(kpt_d[core, :2]).all(axis=1) & (conf_d[core] >= cfg.min_kp_conf)
            good = good_t & good_d
            core_good = core[good]
            if core_good.size >= 2:
                vt = kpt_t[core_good, :2].reshape(-1)
                vd = kpt_d[core_good, :2].reshape(-1)
                if cfg.keypoint_weights is not None:
                    w = np.asarray(cfg.keypoint_weights, dtype=float).reshape(-1)
                    if w.shape[0] == n_k:
                        wg = w[core_good]
                        wg_xy = np.repeat(np.sqrt(np.clip(wg, 0.0, None)), 2)
                        vt = vt * wg_xy
                        vd = vd * wg_xy
                pose_similarity = float(cosine_distance(vt, vd))

    return pose_similarity, pose_dict


def _prepare_pose_for_distance(
    keypoints: Optional[np.ndarray],
    kp_conf: Optional[np.ndarray],
    cfg: MatchConfig,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if keypoints is None:
        return None, None

    kpt = np.asarray(keypoints, dtype=float)
    if kpt.ndim != 2 or kpt.shape[1] < 2:
        return None, None

    kpt = kpt[:, :2]
    n_k = kpt.shape[0]
    conf = (
        np.asarray(kp_conf, dtype=float).reshape(-1)
        if kp_conf is not None else np.ones((n_k,), dtype=float)
    )
    if conf.shape[0] != n_k:
        return None, None

    try:
        out = normalize_pose_2d(kpt, eps=cfg.pose_scale_eps)
        norm_kpt = out[0] if isinstance(out, tuple) else out

    except Exception:
        return None, None

    conf_filtered = conf.copy()
    conf_filtered[conf_filtered < cfg.min_kp_conf] = 0.0

    return norm_kpt, conf_filtered



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
        "tracks": tracks,  # <-- щоб _get_track_id(i) брав trk.track_id
        "detections": detections,  # <-- щоб _get_det_id(j) міг брати id якщо є
        "track_ids": [t.track_id for t in tracks],  # (опційно, але супер-надійно)
        "det_ids": list(range(len(detections)))  # щоб явно було Det#j == j
    }

    log.create_matrix(n_t, n_d)

    if reset_mode:
        track_pose = [(None, None) for _ in tracks]
        det_pose = [(None, None) for _ in detections]


    else:
        track_pose = [
            _prepare_pose_for_distance(trk.last_keypoints, trk.last_kp_conf, cfg)
            for trk in tracks
        ]
        det_pose = [
            _prepare_pose_for_distance(det.keypoints, det.kp_conf, cfg)
            for det in detections
        ]


    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            cell = log[i, j]

            if reset_mode:
                # ignore Kalman gating & motion
                d2 = 0.0
                allowed = True
                d_motion_norm = 0.0

                # ignore pose entirely (neutral)
                d_pose = 0.0
                pose_dict = None
                d_pose_norm = 0.0

            else:
                d_motion, allowed, d2 = _motion_cost_with_gating(trk, det, cfg, gating=True)
                d_motion_norm = float(np.clip(d2 / max(cfg.chi2_gating, 1e-12), 0.0, 1.0))

                trk_kps, trk_conf = track_pose[i]
                det_kps, det_conf = det_pose[j]
                pair_tag = f"Track#{i} (track_id={trk.track_id}) vs Det#{j}"
                d_pose, pose_dict = _pose_distance(
                    trk_kps,
                    trk_conf,
                    det_kps,
                    det_conf,
                    cfg,
                    log=log,
                    pair_tag=pair_tag,
                )
                d_pose_norm = (1.0 - d_pose) / 2.0

            cell.d_motion = float(d_motion_norm)
            cell.allowed = bool(allowed)
            cell.d2 = float(d2)

            if cfg.debug_pose_extended:
                pair_obj = {
                    "track_index": i,
                    "det_index": j,
                    "track_id": trk.track_id,
                    "det_id": j,
                    "pose": pose_dict,
                }
                log.add_pair(pair_obj)

            if not allowed:
                C[i, j] = float(cfg.large_cost)
                cell.cost = float(cfg.large_cost)
                continue

            # appearance (works always)
            if trk.app_emb_ema is not None and isinstance(det.meta, dict) and ("e_app" in det.meta):
                d_app = cosine_distance(trk.app_emb_ema, det.meta["e_app"])
            else:
                d_app = 0.0

            d_app_norm = (1.0 - d_app) / 2.0

            if reset_mode:
                cost = cfg.w_app * d_app_norm
            else:
                cost = (
                        cfg.w_motion * (g * d_motion_norm)
                        + cfg.w_pose * d_pose_norm
                        + cfg.w_app * d_app_norm
                )

            cell.d_pose = float(d_pose_norm)  # або d_pose, як тобі зручніше логувати
            cell.d_app = float(d_app_norm)
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
    """
    Global greedy assignment:
    - build list of all (track, det, cost) where cost < greedy_threshold
      (and also skip large_cost if you still use it as "invalid match")
    - sort by cost ascending
    - take smallest edges while keeping one-to-one constraint
    - return matched pairs + unmatched tracks/dets

    This implements "smaller cost = higher priority" directly.
    """

    thr = greedy_reset_threshold if reset_mode else greedy_threshold


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
            if cost > thr:
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
        reset_mode=reset_mode
    )

    greedy_threshold = cfg.greedy_threshold
    greedy_reset_threshold = cfg.greedy_reset_threshold

    matches, um_tr, um_dt = linear_assignment_with_unmatched(C, large_cost=cfg.large_cost,
        greedy_threshold=greedy_threshold,greedy_reset_threshold=greedy_reset_threshold, reset_mode=reset_mode)
    return matches, um_tr, um_dt, C, log
