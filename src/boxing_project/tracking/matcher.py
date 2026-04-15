from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Union

import numpy as np

from .track import Track, Detection
from . import DEFAULT_TRACKING_CONFIG_PATH
from .tracking_debug import DebugLog
from boxing_project.tracking.normalization import (
    bones_vector,
    mirror_invariant,
    normalize_pose_2d,
)


@dataclass
class MatchConfig:
    alpha: float
    chi2_gating: float
    large_cost: float
    pose_scale_eps: float
    keypoint_weights: Optional[np.ndarray]
    min_kp_conf: float

    greedy_threshold: float
    motion_threshold: float
    pose_threshold: float
    appearance_threshold: float

    emb_ema_alpha: float
    w_motion: float
    w_pose: float
    w_app: float
    min_core_kps: int

    pose_core: list = field(default_factory=list)
    pose_center: list = field(default_factory=list)
    save_log: bool = False

    relative_eps: float = 1e-6
    k_motion: float = 3.0
    k_pose: float = 3.0
    k_app: float = 3.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity in [-1, 1].

    -1: opposite direction
     0: weak/random similarity
     1: identical direction
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _log_ratio_penalty(
    value: float,
    best: float,
    k: float,
    eps: float = 1e-6,
) -> float:
    """
    Relative penalty in [0, 1].

    value == best        -> 0
    value == k * best    -> 1
    value  > k * best    -> 1 (clipped)
    """
    value = float(value)
    best = float(best)
    k = float(max(k, 1.000001))
    eps = float(max(eps, 1e-12))

    ratio = (value + eps) / (best + eps)
    penalty = np.log(ratio) / np.log(k)
    return float(np.clip(penalty, 0.0, 1.0))


def _compute_row_relative_penalties(
    values: np.ndarray,
    valid_mask: np.ndarray,
    k: float,
    eps: float,
) -> np.ndarray:
    """
    values: shape (n_dets,)
    valid_mask: bool mask for valid detections in this row

    Returns penalties in [0, 1].
    Invalid entries get 1.0 by default.
    """
    penalties = np.ones_like(values, dtype=np.float32)

    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        return penalties

    best = float(np.min(values[valid_idx]))

    for j in valid_idx:
        penalties[j] = _log_ratio_penalty(
            value=float(values[j]),
            best=best,
            k=k,
            eps=eps,
        )

    return penalties


def get_common_valid_joints(
    kpt_t: np.ndarray,
    kpt_d: np.ndarray,
    conf_t: Optional[np.ndarray],
    conf_d: Optional[np.ndarray],
    pose_core: np.ndarray,
    min_kp_conf: float,
):
    """
    Returns indices of pose_core joints that are valid in BOTH track and detection.
    """
    if kpt_t is None or kpt_d is None:
        return None
    if kpt_t.ndim != 2 or kpt_d.ndim != 2 or kpt_t.shape[1] < 2 or kpt_d.shape[1] < 2:
        return None
    if kpt_t.shape[0] != kpt_d.shape[0]:
        return None

    n_k = kpt_t.shape[0]

    if conf_t is None:
        conf_t = np.ones((n_k,), dtype=float)
    if conf_d is None:
        conf_d = np.ones((n_k,), dtype=float)

    conf_t = np.asarray(conf_t, dtype=float).reshape(-1)
    conf_d = np.asarray(conf_d, dtype=float).reshape(-1)
    if conf_t.shape[0] != n_k or conf_d.shape[0] != n_k:
        return None

    core = np.asarray(pose_core, dtype=int).reshape(-1)
    core = core[(core >= 0) & (core < n_k)]
    if core.size == 0:
        return None

    good_t = np.isfinite(kpt_t[core, :2]).all(axis=1) & (conf_t[core] >= min_kp_conf)
    good_d = np.isfinite(kpt_d[core, :2]).all(axis=1) & (conf_d[core] >= min_kp_conf)

    return core[good_t & good_d]


def pick_shared_root(kpt_t, kpt_d, conf_t, conf_d, pose_center, min_kp_conf):
    """
    Returns the first common valid index from pose_center.
    If no valid shared keypoint exists, returns None.
    Safe against None inputs.
    """
    if (
        kpt_t is None
        or kpt_d is None
        or conf_t is None
        or conf_d is None
    ):
        return None

    for idx in pose_center:
        if (
            np.isfinite(kpt_t[idx, :2]).all()
            and np.isfinite(kpt_d[idx, :2]).all()
            and conf_t[idx] >= min_kp_conf
            and conf_d[idx] >= min_kp_conf
        ):
            return idx

    return None


def _pose_distance(
    track_keypoints: Optional[np.ndarray],
    track_kp_conf: Optional[np.ndarray],
    det_keypoints: Optional[np.ndarray],
    det_kp_conf: Optional[np.ndarray],
    cfg: MatchConfig,
) -> float:
    """
    Pose cosine similarity from BODY_25 bone direction vectors.
    Uses left/right label-swap mirror invariance for OpenPose robustness.

    Returns similarity in [-1, 1].
    """
    if track_keypoints is None or det_keypoints is None:
        return 0.0

    kpt_t = np.asarray(track_keypoints, dtype=float)
    kpt_d = np.asarray(det_keypoints, dtype=float)

    if kpt_t.ndim != 2 or kpt_d.ndim != 2 or kpt_t.shape[1] < 2 or kpt_d.shape[1] < 2:
        return 0.0
    if kpt_t.shape[0] != kpt_d.shape[0]:
        return 0.0

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
        return 0.0

    bones = np.asarray(
        [(8, 1), (1, 2), (1, 5), (8, 9), (8, 12), (2, 3), (5, 6), (9, 10), (12, 13)],
        dtype=int,
    )

    def _similarity_for_detection(kpt_det: np.ndarray, conf_det: np.ndarray) -> Optional[float]:
        core_good = get_common_valid_joints(
            kpt_t,
            kpt_det,
            conf_t,
            conf_det,
            cfg.pose_core,
            cfg.min_kp_conf,
        )
        if core_good is None or core_good.size < 2:
            return None

        core_set = set(np.asarray(core_good, dtype=int).tolist())
        bones_good = [tuple(b) for b in bones.tolist() if b[0] in core_set and b[1] in core_set]
        if not bones_good:
            return None

        vt_dirs = bones_vector(kpt_t, bones_good)[0]
        vd_dirs = bones_vector(kpt_det, bones_good)[0]
        return float(cosine_similarity(vt_dirs.reshape(-1), vd_dirs.reshape(-1)))

    sims = []

    sim_normal = _similarity_for_detection(kpt_d, conf_d)
    if sim_normal is not None:
        sims.append(sim_normal)

    kpt_d_m, conf_d_m = mirror_invariant(kpt_d, conf_d)
    sim_mirror = _similarity_for_detection(kpt_d_m, conf_d_m)
    if sim_mirror is not None:
        sims.append(sim_mirror)

    if not sims:
        return 0.0

    return float(max(sims))


def _prepare_pose_for_distance(keypoints, kp_conf, cfg):
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

    conf[conf < cfg.min_kp_conf] = np.nan
    return kpt, conf


def _motion_cost_with_gating(
    track: Track,
    det: Detection,
    cfg: MatchConfig,
    gating: bool,
) -> Tuple[float, bool, float]:
    d2 = track.kf.gating_distance(np.asarray(det.center, dtype=float))
    allowed = d2 <= cfg.chi2_gating
    d_motion = float(np.sqrt(max(d2, 0.0)))

    if gating is False:
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
        "tracks": tracks,
        "detections": detections,
        "track_ids": [t.track_id for t in tracks],
        "det_ids": list(range(len(detections))),
    }

    log.create_matrix(n_t, n_d)

    track_pose = [
        _prepare_pose_for_distance(trk.last_keypoints, trk.last_kp_conf, cfg)
        for trk in tracks
    ]
    det_pose = [
        _prepare_pose_for_distance(det.keypoints, det.kp_conf, cfg)
        for det in detections
    ]

    for i, trk in enumerate(tracks):
        row_motion = np.full((n_d,), np.inf, dtype=np.float32)
        row_pose = np.full((n_d,), np.inf, dtype=np.float32)
        row_app = np.full((n_d,), np.inf, dtype=np.float32)
        row_valid = np.zeros((n_d,), dtype=bool)

        # PASS 1: raw pairwise values + hard gating
        for j, det in enumerate(detections):
            cell = log[i, j]

            d_motion, allowed, d2 = _motion_cost_with_gating(
                trk, det, cfg, gating=True
            )
            d_motion_norm = float(
                np.clip(d2 / max(cfg.chi2_gating, 1e-12), 0.0, 1.0)
            )

            trk_kps, trk_conf = track_pose[i]
            det_kps, det_conf = det_pose[j]

            root_idx = pick_shared_root(
                trk_kps,
                det_kps,
                trk_conf,
                det_conf,
                cfg.pose_center,
                cfg.min_kp_conf,
            )

            if root_idx is not None:
                trk_kps = normalize_pose_2d(trk_kps, root_idx)
                det_kps = normalize_pose_2d(det_kps, root_idx)

            sim_pose = _pose_distance(trk_kps, trk_conf, det_kps, det_conf, cfg)
            pose_cost = float((1.0 - sim_pose) / 2.0)

            if trk.app_emb_ema is not None and isinstance(det.meta, dict) and ("e_app" in det.meta):
                sim_app = cosine_similarity(trk.app_emb_ema, det.meta["e_app"])
            else:
                sim_app = 0.0

            app_cost = float((1.0 - sim_app) / 2.0)

            motion_ok = d_motion_norm <= cfg.motion_threshold
            pose_ok = pose_cost <= cfg.pose_threshold
            app_ok = app_cost <= cfg.appearance_threshold

            valid = bool(allowed and motion_ok and pose_ok and app_ok)

            cell.d_motion = float(d_motion_norm)
            cell.allowed = bool(allowed)
            cell.d2 = float(d2)
            cell.d_pose = float(pose_cost)
            cell.d_app = float(app_cost)
            cell.motion_ok = bool(motion_ok)
            cell.pose_ok = bool(pose_ok)
            cell.app_ok = bool(app_ok)

            row_motion[j] = float(d_motion_norm)
            row_pose[j] = float(pose_cost)
            row_app[j] = float(app_cost)
            row_valid[j] = valid

        # PASS 2: row-wise relative penalties
        rel_motion = _compute_row_relative_penalties(
            values=row_motion,
            valid_mask=row_valid,
            k=cfg.k_motion,
            eps=cfg.relative_eps,
        )
        rel_pose = _compute_row_relative_penalties(
            values=row_pose,
            valid_mask=row_valid,
            k=cfg.k_pose,
            eps=cfg.relative_eps,
        )
        rel_app = _compute_row_relative_penalties(
            values=row_app,
            valid_mask=row_valid,
            k=cfg.k_app,
            eps=cfg.relative_eps,
        )

        # PASS 3: final RELATIVE-ONLY cost
        for j in range(n_d):
            cell = log[i, j]

            if not row_valid[j]:
                C[i, j] = float(cfg.large_cost)
                cell.cost = float(cfg.large_cost)
                cell.rel_motion = 1.0
                cell.rel_pose = 1.0
                cell.rel_app = 1.0
                continue

            cell.rel_motion = float(rel_motion[j])
            cell.rel_pose = float(rel_pose[j])
            cell.rel_app = float(rel_app[j])

            cost = (
                cfg.w_motion * float(rel_motion[j])
                + cfg.w_pose * float(rel_pose[j])
                + cfg.w_app * float(rel_app[j])
            )

            cell.cost = float(cost)
            C[i, j] = float(cost)

    if show or cfg.save_log:
        log.show_matrix()

    return C, log


def linear_assignment_with_unmatched(
    C: np.ndarray,
    large_cost: float,
    greedy_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if C.size == 0:
        return [], list(range(C.shape[0])), list(range(C.shape[1]))

    n_tracks, n_dets = C.shape

    edges: List[Tuple[float, int, int]] = []
    for r in range(n_tracks):
        for c in range(n_dets):
            cost = float(C[r, c])
            if cost >= large_cost:
                continue
            if cost > greedy_threshold:
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


def _load_match_config_from_yaml(
    config_path: Optional[Union[str, Path]] = None,
) -> MatchConfig:
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
    )

    matches, um_tr, um_dt = linear_assignment_with_unmatched(
        C=C,
        large_cost=cfg.large_cost,
        greedy_threshold=cfg.greedy_threshold,
    )

    return matches, um_tr, um_dt, C, log