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


# ----------------------------- #
#         Match config          #
# ----------------------------- #

# src/boxing_project/tracking/matcher.py  (у MatchConfig додай поля)

@dataclass
class MatchConfig:
    """"Конфіг для побудови cost matrix (motion/pose/app + gating)." """
    alpha: float
    chi2_gating: float
    large_cost: float
    pose_scale_eps: float
    keypoint_weights: Optional[np.ndarray]
    min_kp_conf: float


    w_motion_base: float = 1.0
    w_pose_base: float = 1.0
    w_pose_cut: float = 2.0
    w_app_base: float = 1.0
    w_app_cut: float = 3.0
    emb_ema_alpha: float = 0.9


# ----------------------------- #
#        Pose helpers           #
# ----------------------------- #


def _pose_distance(
    track: Track,
    det: Detection,
    cfg: MatchConfig,
    log: Optional[DebugLog] = None,
    pair_tag: str = ""
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute pose-based distance between a track and a detection.

    Assumptions in this simplified version:
      - Input keypoints are already normalized (no centering / scaling here).
      - All joints have the same weight (no per-joint weighting).
      - Confidences are used only to select "good" joints (>= min_kp_conf),
        not as weights in the final average.

    Steps:
      1) Take last keypoints from the track and current keypoints from the detection.
      2) Drop joints with low confidence or NaNs.
      3) Directly compute per-joint Euclidean distance on used joints.
      4) Return simple mean distance and a dict with detailed debug information.

    Returns:
      D_pose : float
          Final pose distance between track and detection.
      pose_dict : Dict[str, Any]
          Detailed information for debugging / logging.
    """
    pose_dict: Dict[str, Any] = {}

    # No keypoints → nothing to compare
    if track.last_keypoints is None or det.keypoints is None:
        set_pose_no_keypoints(pose_dict, log, pair_tag)
        return 0.0, pose_dict

    kpt_t = np.asarray(track.last_keypoints, dtype=float)  # (K, 2)
    kpt_d = np.asarray(det.keypoints, dtype=float)         # (K, 2)
    n_k = kpt_t.shape[0]

    # Confidences for filtering; if missing → ones
    conf_t = (
        np.asarray(track.last_kp_conf, dtype=float).reshape(-1)
        if track.last_kp_conf is not None
        else np.ones((n_k,), dtype=float)
    )
    conf_d = (
        np.asarray(det.kp_conf, dtype=float).reshape(-1)
        if det.kp_conf is not None
        else np.ones((n_k,), dtype=float)
    )

    assert kpt_t.shape[0] == kpt_d.shape[0], "Track/Det must have same number of keypoints"

    # "Good" joints: finite coords + conf >= min_kp_conf on both sides
    good_t = np.isfinite(kpt_t).all(axis=1) & (conf_t >= cfg.min_kp_conf)
    good_d = np.isfinite(kpt_d).all(axis=1) & (conf_d >= cfg.min_kp_conf)
    good = good_t & good_d

    if not np.any(good):
        # No reliable joints for this pair
        set_pose_no_good_points(pose_dict, log, pair_tag, good)
        return 0.0, pose_dict

    # No normalization: we assume kpt_t and kpt_d already in a comparable space
    kpt_t_used = kpt_t[good]
    kpt_d_used = kpt_d[good]

    diff_used = kpt_t_used - kpt_d_used           # (N_good, 2)
    per_k_used = np.linalg.norm(diff_used, axis=1)  # (N_good,)

    # All joints have equal weight → simple mean over used joints
    w_used = np.ones_like(per_k_used, dtype=float)
    D_pose = float(per_k_used.mean())

    # For debug we still call fill_pose_full_debug, passing original coordinates
    fill_pose_full_debug(
        pose_dict=pose_dict,
        log=log,
        pair_tag=pair_tag,
        n_k=n_k,
        good_mask=good,
        kpt_tn=kpt_t,          # not normalized, but used as-is for logging
        kpt_dn=kpt_d,          # same
        diff_used=diff_used,
        per_k_used=per_k_used,
        w_used=w_used,
        D_pose=D_pose,
    )

    return D_pose, pose_dict


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance = 1 - cos(a,b).

    Повертає 1.0, якщо хоч один вектор нульовий/порожній,
    щоб уникнути ділення на 0.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(1.0 - float(np.dot(a, b) / (na * nb)))


# ----------------------------- #
#       Motion + gating         #
# ----------------------------- #

def _motion_cost_with_gating(
    track: Track,
    det: Detection,
    cfg: MatchConfig,
    log: Optional[DebugLog] = None,
    pair_tag: str = ""
) -> Tuple[float, bool, float]:
    """
    Compute motion-based cost between a track and a detection using Kalman gating.

    Steps:
      1) Use KalmanTracker.gating_distance() to get Mahalanobis distance^2 (d2)
         between predicted state and detection center.
      2) Compare d2 with chi2_gating threshold:
         - if d2 > chi2_gating -> pair is NOT allowed (pruned).
         - else -> pair is allowed.
      3) Motion cost = sqrt(d2) (for allowed pairs; still defined for logging).

    Returns:
      d_motion : float
          Motion distance (sqrt of d2).
      allowed : bool
          Whether this pair passes the χ² gating.
      d2 : float
          Raw Mahalanobis distance squared (for debug).
    """
    d2 = track.kf.gating_distance(np.asarray(det.center, dtype=float))
    allowed = (d2 <= cfg.chi2_gating)
    d_motion = float(np.sqrt(max(d2, 0.0)))

    if log and log.enabled_print:
        log.section(f"[{pair_tag}] MOTION")
        check = "✓" if allowed else "✗"
        log._print(f"• d² = {d2:.6f}   |   χ²_gate = {cfg.chi2_gating:.6f}   |   allowed = {check}")
        log._print(f"• d_motion = √(d²) = {d_motion:.6f}")

    return d_motion, allowed, float(d2)


# ----------------------------- #
#        Cost matrix C          #
# ----------------------------- #

def build_cost_matrix(
    tracks: List[Track],
    detections: List[Detection],
    cfg: MatchConfig,
    show: bool = True,
    sink: Optional[Callable[[str], None]] = None,
    g: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Будує cost matrix для Hungarian.

    Формула:
        cost = w_motion(g) * d_motion
             + w_pose(g)   * d_pose
             + w_app(g)    * d_app

    Де:
      - d_motion: sqrt(Mahalanobis^2) з Kalman gating (через _motion_cost_with_gating)
      - d_pose: cosine distance між track.pose_emb_ema та det.meta["e_pose"]
                fallback: _pose_distance() по keypoints
      - d_app: cosine distance між track.app_emb_ema та det.meta["e_app"]
               якщо нема embedding -> 0.0

    Повертає:
      - C: (n_tracks, n_dets)
      - log_matcher: dict з ключем "pairs" (щоб tracker.py міг зібрати pair logs)
    """
    n_t = len(tracks)
    n_d = len(detections)
    C = np.zeros((n_t, n_d), dtype=np.float32)

    # clamp g
    g = float(max(0.0, min(1.0, g)))

    # ваги
    w_motion = g * float(cfg.w_motion_base)
    w_pose = (1.0 - g) * float(cfg.w_pose_cut) + g * float(cfg.w_pose_base)
    w_app = (1.0 - g) * float(cfg.w_app_cut) + g * float(cfg.w_app_base)

    # Мінімальний лог у форматі, який очікує tracker.py
    log_matcher: Dict[str, Any] = {
        "g": g,
        "weights": {"w_motion": w_motion, "w_pose": w_pose, "w_app": w_app},
        "shape": [n_t, n_d],
        "pairs": []
    }

    # debug log (твій DebugLog), але можна не вмикати
    log = create_matcher_log(cfg, (n_t, n_d), show=show, sink=sink)

    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            pair_tag = f"{i}-{j}"

            # motion + gating
            d_motion, allowed, d2 = _motion_cost_with_gating(trk, det, cfg, log=log, pair_tag=pair_tag)

            pair_obj = make_pair_base(track_index=i, det_index=j)
            if not allowed:
                C[i, j] = float(cfg.large_cost)
                fill_pair_gated_out(
                    pair_obj=pair_obj,
                    cfg=cfg,
                    d2=d2,
                    d_motion=d_motion,
                    cost=float(cfg.large_cost),
                )
                log_matcher["pairs"].append(pair_obj)
                if log.enabled_print:
                    print_gating_result(log, pair_tag, allowed, d2, cfg.chi2_gating, cfg.large_cost)
                continue

            # pose: embedding або fallback keypoints
            d_pose_raw, pose_dict = _pose_distance(trk, det, cfg, log=log, pair_tag=pair_tag)

            if trk.pose_emb_ema is not None and "e_pose" in det.meta:
                d_pose = cosine_distance(trk.pose_emb_ema, det.meta["e_pose"])
                pose_reason = "pose_embedding"
            else:
                d_pose = float(d_pose_raw)
                pose_reason = "pose_keypoints_fallback"

            # appearance: embedding або 0
            if trk.app_emb_ema is not None and "e_app" in det.meta:
                d_app = cosine_distance(trk.app_emb_ema, det.meta["e_app"])
                app_reason = "app_embedding"
            else:
                d_app = 0.0
                app_reason = "no_app_embedding"

            # final cost
            cost = w_motion * float(d_motion) + w_pose * float(d_pose) + w_app * float(d_app)
            C[i, j] = float(cost)

            # log pair
            fill_pair_ok(
                pair_obj=pair_obj,
                cfg=cfg,
                d_motion=float(d_motion),
                d_pose=float(d_pose),
                cost=float(cost),
                pose_dict=pose_dict,
            )
            # доповнимо інфо, щоб було видно нові компоненти
            pair_obj["final"]["components"]["d_app"] = float(d_app)
            pair_obj["final"]["components"]["w_motion"] = float(w_motion)
            pair_obj["final"]["components"]["w_pose"] = float(w_pose)
            pair_obj["final"]["components"]["w_app"] = float(w_app)
            pair_obj["final"]["components"]["g"] = float(g)
            pair_obj["final"]["reason"] = f"{pose_reason}+{app_reason}"

            log_matcher["pairs"].append(pair_obj)

            if log.enabled_print:
                print_pair_result(log, pair_tag, pair_obj, d_motion, d_pose, C[i, j])

    return C, log_matcher



# ----------------------------- #
#      Hungarian + unmatched    #
# ----------------------------- #

def linear_assignment_with_unmatched(
    C: np.ndarray,
    large_cost: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Run Hungarian algorithm on cost matrix and also return unmatched indices.

    Steps:
      1) Run linear_sum_assignment on C.
      2) Filter out assignments whose cost >= large_cost (treated as invalid).
      3) Build:
         - matched : list of (track_idx, det_idx) for valid assignments.
         - unmatched_tracks : track indices not present in matched.
         - unmatched_dets   : detection indices not present in matched.

    Parameters:
      C : np.ndarray
          Cost matrix (tracks x detections).
      large_cost : float
          Threshold for "invalid" matches (e.g. gated out or too expensive).

    Returns:
      matched : List[Tuple[int, int]]
      unmatched_tracks : List[int]
      unmatched_dets : List[int]
    """
    if C.size == 0:
        return [], list(range(C.shape[0])), list(range(C.shape[1]))

    rows, cols = linear_sum_assignment(C)

    matched: List[Tuple[int, int]] = []
    used_tracks = set()
    used_dets = set()

    for r, c in zip(rows, cols):
        if C[r, c] >= large_cost:
            # This match is effectively forbidden (gated out or too costly).
            continue
        matched.append((int(r), int(c)))
        used_tracks.add(int(r))
        used_dets.add(int(c))

    n_tracks, n_dets = C.shape
    unmatched_tracks = [i for i in range(n_tracks) if i not in used_tracks]
    unmatched_dets = [j for j in range(n_dets) if j not in used_dets]

    return matched, unmatched_tracks, unmatched_dets


# ----------------------------- #
#        Config loading         #
# ----------------------------- #

def _load_match_config_from_yaml(
    config_path: Optional[Union[str, Path]] = None,
) -> MatchConfig:
    """
    Read MatchConfig from a YAML file via utils.config helpers.

    If config_path is None, DEFAULT_TRACKING_CONFIG_PATH is used.
    """
    from boxing_project.utils.config import load_tracking_config

    resolved = Path(config_path) if config_path is not None else DEFAULT_TRACKING_CONFIG_PATH
    # load_tracking_config already creates the dataclass instances.
    _, match_cfg, _ = load_tracking_config(str(resolved))
    # Copy to avoid sharing mutable state across callers.
    return copy.deepcopy(match_cfg)


# ----------------------------- #
#    Public matching function   #
# ----------------------------- #

def match_tracks_and_detections(
        tracks,
        detections,
        cfg=None,
        debug=True,
        sink=None,
        config_path=None,
        g: float = 1.0,
):
    """
    High-level API: match existing tracks with current detections.

    If cfg is None:
      - Load MatchConfig from YAML (config_path or default).

    Then:
      1) Build cost matrix C via build_cost_matrix().
      2) Run Hungarian matching with linear_assignment_with_unmatched().
      3) Return matches and debug information.

    Returns:
      matches : List[Tuple[int, int]]
          (track_index, detection_index) pairs.
      um_tr   : List[int]
          Indices of unmatched tracks.
      um_dt   : List[int]
          Indices of unmatched detections.
      C       : np.ndarray
          Final cost matrix used for Hungarian.
      log_matcher : Dict[str, Any]
          Detailed matcher log for this frame (for debugging / visualization).

     g ∈ [0,1] — довіра до motion:
    - g=1: motion домінує (звичайні кадри)
    - g=0: motion гаситься, embeddings домінують (camera cut)
    """
    if cfg is None:
        cfg = _load_match_config_from_yaml(config_path)

    C, log_matcher = build_cost_matrix(
        tracks=tracks,
        detections=detections,
        cfg=cfg,
        show=debug,
        sink=sink,
        g=g
    )
    matches, um_tr, um_dt = linear_assignment_with_unmatched(C, cfg.large_cost)
    return matches, um_tr, um_dt, C, log_matcher
