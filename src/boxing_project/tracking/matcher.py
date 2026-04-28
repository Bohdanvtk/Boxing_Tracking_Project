from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .track import Detection, Track
from . import DEFAULT_TRACKING_CONFIG_PATH
from .tracking_debug import DebugLog
from boxing_project.tracking.normalization import (
    bones_vector,
    mirror_invariant,
    normalize_pose_2d,
)


# ============================================================================
# Configuration / data containers
# ============================================================================

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


@dataclass
class PairwiseArtifacts:
    """
    Static pairwise data computed once for the current frame.
    These values are reused both for row-ranking and for column conflict resolution.
    """
    motion: np.ndarray         # shape (n_tracks, n_dets)
    pose: np.ndarray           # shape (n_tracks, n_dets)
    app: np.ndarray            # shape (n_tracks, n_dets)
    valid: np.ndarray          # shape (n_tracks, n_dets), bool

    row_cost: np.ndarray       # row-relative preference cost matrix
    row_preferences: List[List[int]]

    log: DebugLog


# ============================================================================
# Low-level math helpers
# ============================================================================

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

    value == best     -> 0
    value == k*best   -> 1
    value >  k*best   -> 1 (clipped)
    """
    value = float(value)
    best = float(best)
    k = float(max(k, 1.000001))
    eps = float(max(eps, 1e-12))

    ratio = (value + eps) / (best + eps)
    penalty = np.log(ratio) / np.log(k)
    return float(np.clip(penalty, 0.0, 1.0))


def _compute_subset_relative_penalties(
    values: np.ndarray,
    active_mask: np.ndarray,
    k: float,
    eps: float,
) -> np.ndarray:
    """
    Compute relative penalties over an arbitrary subset along one axis.

    Parameters
    ----------
    values:
        1D array of raw distances.
    active_mask:
        Boolean mask that marks which entries participate in the comparison.
    k:
        Saturation ratio for log-ratio penalty.
    eps:
        Numerical epsilon.

    Returns
    -------
    penalties:
        1D float array in [0, 1].
        Inactive entries get 1.0 by default.
    """
    penalties = np.ones_like(values, dtype=np.float32)

    active_idx = np.where(active_mask)[0]
    if active_idx.size == 0:
        return penalties

    best = float(np.min(values[active_idx]))
    for idx in active_idx:
        penalties[idx] = _log_ratio_penalty(
            value=float(values[idx]),
            best=best,
            k=k,
            eps=eps,
        )

    return penalties


# ============================================================================
# Pose helpers
# ============================================================================

def get_common_valid_joints(
    kpt_t: np.ndarray,
    kpt_d: np.ndarray,
    conf_t: Optional[np.ndarray],
    conf_d: Optional[np.ndarray],
    pose_core: np.ndarray,
    min_kp_conf: float,
):
    """
    Return indices of pose_core joints that are valid in BOTH track and detection.
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


def pick_shared_root(
    kpt_t,
    kpt_d,
    conf_t,
    conf_d,
    pose_center,
    min_kp_conf,
):
    """
    Return the first common valid index from pose_center.
    If no valid shared keypoint exists, return None.
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

    def _similarity_for_detection(
        kpt_det: np.ndarray,
        conf_det: np.ndarray,
    ) -> Optional[float]:
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
        bones_good = [
            tuple(b) for b in bones.tolist()
            if b[0] in core_set and b[1] in core_set
        ]
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


# ============================================================================
# Pairwise feature computation
# ============================================================================

def _motion_cost_with_gating(
    track: Track,
    det: Detection,
    cfg: MatchConfig,
    gating: bool,
) -> Tuple[float, bool, float]:
    """
    Return:
      - sqrt(chi2 distance)
      - allowed by chi2 gate
      - raw chi2 distance
    """
    d2 = track.kf.gating_distance(np.asarray(det.center, dtype=float))
    allowed = d2 <= cfg.chi2_gating
    d_motion = float(np.sqrt(max(d2, 0.0)))

    if gating is False:
        allowed = True

    return d_motion, allowed, float(d2)


def _compute_pairwise_components(
    track: Track,
    det: Detection,
    track_pose_item,
    det_pose_item,
    cfg: MatchConfig,
) -> Tuple[float, float, float, bool, float, bool, bool, bool]:
    """
    Compute raw pairwise values with early-exit gating.

    Order:
      1) motion / chi2
      2) pose
      3) appearance

    If a pair fails earlier gates, later expensive components are skipped.
    """
    _, allowed, d2 = _motion_cost_with_gating(track, det, cfg, gating=True)
    d_motion_norm = float(np.clip(d2 / max(cfg.chi2_gating, 1e-12), 0.0, 1.0))

    motion_ok = d_motion_norm <= cfg.motion_threshold

    # Early exit:
    # If motion is already invalid, do not compute pose/app.
    if not allowed or not motion_ok:
        return (
            d_motion_norm,
            1.0,          # pose_cost fallback
            1.0,          # app_cost fallback
            bool(allowed),
            float(d2),
            bool(motion_ok),
            False,        # pose_ok
            False,        # app_ok
        )

    trk_kps, trk_conf = track_pose_item
    det_kps, det_conf = det_pose_item

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
    pose_ok = pose_cost <= cfg.pose_threshold

    # Early exit:
    # If pose is invalid, do not compute appearance.
    if not pose_ok:
        return (
            d_motion_norm,
            pose_cost,
            1.0,          # app_cost fallback
            bool(allowed),
            float(d2),
            bool(motion_ok),
            bool(pose_ok),
            False,        # app_ok
        )

    sim_app = 0.0

    if track.app_emb_ema is not None and isinstance(det.meta, dict) and ("e_app" in det.meta):
        det_emb = det.meta.get("e_app", None)

        if det_emb is not None:
            track_emb = np.asarray(track.app_emb_ema, dtype=np.float32).reshape(-1)
            det_emb = np.asarray(det_emb, dtype=np.float32).reshape(-1)

            if track_emb.shape[0] == det_emb.shape[0]:
                sim_app = cosine_similarity(track_emb, det_emb)

    app_cost = float((1.0 - sim_app) / 2.0)
    app_ok = app_cost <= cfg.appearance_threshold

    return (
        d_motion_norm,
        pose_cost,
        app_cost,
        bool(allowed),
        float(d2),
        bool(motion_ok),
        bool(pose_ok),
        bool(app_ok),
    )


def _initialize_debug_log(
    tracks: List[Track],
    detections: List[Detection],
    show: bool,
    sink: Optional[Callable[[str], None]],
    g: float,
) -> DebugLog:
    log = DebugLog(enabled_print=show, sink=sink or print)
    log.meta = {
        "g": g,
        "tracks": tracks,
        "detections": detections,
        "track_ids": [t.track_id for t in tracks],
        "det_ids": list(range(len(detections))),
    }
    log.create_matrix(len(tracks), len(detections))
    return log


def _compute_pairwise_matrices(
    tracks: List[Track],
    detections: List[Detection],
    cfg: MatchConfig,
    log: DebugLog,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute raw pairwise matrices once.

    Returns
    -------
    motion, pose, app, valid
    """
    n_t = len(tracks)
    n_d = len(detections)

    motion = np.full((n_t, n_d), np.inf, dtype=np.float32)
    pose = np.full((n_t, n_d), np.inf, dtype=np.float32)
    app = np.full((n_t, n_d), np.inf, dtype=np.float32)
    valid = np.zeros((n_t, n_d), dtype=bool)

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

            (
                d_motion_norm,
                pose_cost,
                app_cost,
                allowed,
                d2,
                motion_ok,
                pose_ok,
                app_ok,
            ) = _compute_pairwise_components(
                track=trk,
                det=det,
                track_pose_item=track_pose[i],
                det_pose_item=det_pose[j],
                cfg=cfg,
            )

            is_valid = bool(allowed and motion_ok and pose_ok and app_ok)

            motion[i, j] = float(d_motion_norm)
            pose[i, j] = float(pose_cost)
            app[i, j] = float(app_cost)
            valid[i, j] = is_valid

            cell.d_motion = float(d_motion_norm)
            cell.d_pose = float(pose_cost)
            cell.d_app = float(app_cost)
            cell.allowed = bool(allowed)
            cell.d2 = float(d2)
            cell.motion_ok = bool(motion_ok)
            cell.pose_ok = bool(pose_ok)
            cell.app_ok = bool(app_ok)

    return motion, pose, app, valid


# ============================================================================
# Row-relative preference construction
# ============================================================================

def _compute_row_relative_costs(
    motion: np.ndarray,
    pose: np.ndarray,
    app: np.ndarray,
    valid: np.ndarray,
    cfg: MatchConfig,
    log: DebugLog,
) -> np.ndarray:
    """
    Build static row-relative cost matrix.

    This matrix defines per-track detection preference order.
    It is NOT the final assignment decision by itself.
    """
    n_t, n_d = motion.shape
    row_cost = np.full((n_t, n_d), cfg.large_cost, dtype=np.float32)

    for i in range(n_t):
        row_valid = valid[i, :]
        rel_motion = _compute_subset_relative_penalties(
            values=motion[i, :],
            active_mask=row_valid,
            k=cfg.k_motion,
            eps=cfg.relative_eps,
        )
        rel_pose = _compute_subset_relative_penalties(
            values=pose[i, :],
            active_mask=row_valid,
            k=cfg.k_pose,
            eps=cfg.relative_eps,
        )
        rel_app = _compute_subset_relative_penalties(
            values=app[i, :],
            active_mask=row_valid,
            k=cfg.k_app,
            eps=cfg.relative_eps,
        )

        for j in range(n_d):
            cell = log[i, j]

            if not row_valid[j]:
                cell.rel_motion = 1.0
                cell.rel_pose = 1.0
                cell.rel_app = 1.0
                cell.cost = float(cfg.large_cost)
                continue

            cost = (
                cfg.w_motion * float(rel_motion[j])
                + cfg.w_pose * float(rel_pose[j])
                + cfg.w_app * float(rel_app[j])
            )

            row_cost[i, j] = float(cost)

            cell.rel_motion = float(rel_motion[j])
            cell.rel_pose = float(rel_pose[j])
            cell.rel_app = float(rel_app[j])
            cell.cost = float(cost)

    return row_cost


def _build_row_preferences(
    row_cost: np.ndarray,
    valid: np.ndarray,
    greedy_threshold: float,
) -> List[List[int]]:
    """
    For each track, build an ordered list of candidate detections.

    Preference order is determined by row-relative cost (lower is better).
    Only valid pairs under greedy_threshold are included.
    """
    n_t, _ = row_cost.shape
    row_preferences: List[List[int]] = []

    for i in range(n_t):
        candidate_js = [
            j
            for j in np.where(valid[i])[0].tolist()
            if float(row_cost[i, j]) <= float(greedy_threshold)
        ]
        ordered = sorted(candidate_js, key=lambda j: float(row_cost[i, j]))
        row_preferences.append(ordered)

    return row_preferences


def build_cost_matrix(
    tracks: List[Track],
    detections: List[Detection],
    cfg: MatchConfig,
    show: bool = True,
    sink: Optional[Callable[[str], None]] = None,
    g: float = 1.0,
) -> Tuple[PairwiseArtifacts, np.ndarray]:
    """
    Build static matching artifacts.

    Returns
    -------
    artifacts:
        Full static data for the current frame.
    row_cost:
        Row-relative cost matrix (useful for inspection / return value).
    """
    g = float(np.clip(g, 0.0, 1.0))
    log = _initialize_debug_log(
        tracks=tracks,
        detections=detections,
        show=show,
        sink=sink,
        g=g,
    )

    motion, pose, app, valid = _compute_pairwise_matrices(
        tracks=tracks,
        detections=detections,
        cfg=cfg,
        log=log,
    )

    row_cost = _compute_row_relative_costs(
        motion=motion,
        pose=pose,
        app=app,
        valid=valid,
        cfg=cfg,
        log=log,
    )

    row_preferences = _build_row_preferences(
        row_cost=row_cost,
        valid=valid,
        greedy_threshold=cfg.greedy_threshold,
    )

    artifacts = PairwiseArtifacts(
        motion=motion,
        pose=pose,
        app=app,
        valid=valid,
        row_cost=row_cost,
        row_preferences=row_preferences,
        log=log,
    )

    if show or cfg.save_log:
        log.show_matrix()

    return artifacts, row_cost


# ============================================================================
# Column conflict resolution
# ============================================================================

def _compute_column_conflict_costs(
    det_idx: int,
    candidate_tracks: Sequence[int],
    artifacts: PairwiseArtifacts,
    cfg: MatchConfig,
) -> Dict[int, float]:
    """
    Resolve a conflict on one detection using column-relative comparison.

    Parameters
    ----------
    det_idx:
        Detection column under conflict.
    candidate_tracks:
        Tracks currently competing for this detection.
    artifacts:
        Static pairwise data.
    cfg:
        Matching config.

    Returns
    -------
    dict(track_idx -> column_conflict_cost)
    """
    cand = np.asarray(list(candidate_tracks), dtype=int)
    if cand.size == 0:
        return {}

    active = np.ones((cand.size,), dtype=bool)

    rel_motion = _compute_subset_relative_penalties(
        values=artifacts.motion[cand, det_idx],
        active_mask=active,
        k=cfg.k_motion,
        eps=cfg.relative_eps,
    )
    rel_pose = _compute_subset_relative_penalties(
        values=artifacts.pose[cand, det_idx],
        active_mask=active,
        k=cfg.k_pose,
        eps=cfg.relative_eps,
    )
    rel_app = _compute_subset_relative_penalties(
        values=artifacts.app[cand, det_idx],
        active_mask=active,
        k=cfg.k_app,
        eps=cfg.relative_eps,
    )

    result: Dict[int, float] = {}
    for local_idx, track_idx in enumerate(cand.tolist()):
        col_cost = (
            cfg.w_motion * float(rel_motion[local_idx])
            + cfg.w_pose * float(rel_pose[local_idx])
            + cfg.w_app * float(rel_app[local_idx])
        )
        result[int(track_idx)] = float(col_cost)

    return result


def _choose_detection_owner(
    det_idx: int,
    candidate_tracks: Sequence[int],
    artifacts: PairwiseArtifacts,
    cfg: MatchConfig,
) -> int:
    """
    Choose the owner of one detection among current candidate tracks.

    Primary criterion:
        column-relative conflict cost

    Tie-break:
        lower static row-relative cost
        then smaller track index for determinism
    """
    conflict_costs = _compute_column_conflict_costs(
        det_idx=det_idx,
        candidate_tracks=candidate_tracks,
        artifacts=artifacts,
        cfg=cfg,
    )

    def _sort_key(track_idx: int) -> Tuple[float, float, int]:
        return (
            float(conflict_costs[int(track_idx)]),
            float(artifacts.row_cost[int(track_idx), int(det_idx)]),
            int(track_idx),
        )

    return min(candidate_tracks, key=_sort_key)


# ============================================================================
# Final assignment algorithm
# ============================================================================

def deferred_match_with_column_resolution(
    artifacts: PairwiseArtifacts,
    cfg: MatchConfig,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Iterative one-to-one matching with:

      1. row-relative preference lists per track
      2. column-relative conflict resolution per detection

    High-level logic
    ----------------
    - each track proposes to its best remaining detection by row preference
    - if several tracks want the same detection, that detection chooses the
      best owner by column-relative comparison among the current contenders
    - rejected tracks move to their next-best detection
    - process repeats until no free track can propose anymore
    """
    n_tracks, n_dets = artifacts.row_cost.shape

    next_choice_idx = [0] * n_tracks
    det_owner: Dict[int, int] = {}

    free_tracks = {
        i for i in range(n_tracks)
        if len(artifacts.row_preferences[i]) > 0
    }

    exhausted_tracks = set()

    while free_tracks:
        proposals: Dict[int, List[int]] = {}

        # Each free track proposes to its next preferred detection.
        for track_idx in list(free_tracks):
            prefs = artifacts.row_preferences[track_idx]

            if next_choice_idx[track_idx] >= len(prefs):
                exhausted_tracks.add(track_idx)
                continue

            det_idx = prefs[next_choice_idx[track_idx]]
            proposals.setdefault(det_idx, []).append(track_idx)

        # We rebuild free set from losers of this round.
        free_tracks = set()

        for det_idx, proposing_tracks in proposals.items():
            contenders = list(proposing_tracks)

            # Incumbent owner also participates in the comparison.
            if det_idx in det_owner:
                incumbent = det_owner[det_idx]
                if incumbent not in contenders:
                    contenders.append(incumbent)

            winner = _choose_detection_owner(
                det_idx=det_idx,
                candidate_tracks=contenders,
                artifacts=artifacts,
                cfg=cfg,
            )
            det_owner[det_idx] = winner

            for loser in contenders:
                if loser == winner:
                    continue

                next_choice_idx[loser] += 1
                if next_choice_idx[loser] < len(artifacts.row_preferences[loser]):
                    free_tracks.add(loser)
                else:
                    exhausted_tracks.add(loser)

    matches = sorted((track_idx, det_idx) for det_idx, track_idx in det_owner.items())

    matched_tracks = {i for i, _ in matches}
    matched_dets = {j for _, j in matches}

    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
    unmatched_dets = [j for j in range(n_dets) if j not in matched_dets]

    return matches, unmatched_tracks, unmatched_dets


# ============================================================================
# Config loading / public API
# ============================================================================

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
    """
    Public entry point used by the tracker.

    Notes
    -----
    `reset_mode` is preserved in the signature for compatibility with the
    surrounding pipeline, but this matcher currently does not branch on it.
    """
    if cfg is None:
        cfg = _load_match_config_from_yaml(config_path)

    artifacts, row_cost = build_cost_matrix(
        tracks=tracks,
        detections=detections,
        cfg=cfg,
        show=debug,
        sink=sink,
        g=g,
    )

    matches, um_tr, um_dt = deferred_match_with_column_resolution(
        artifacts=artifacts,
        cfg=cfg,
    )

    return matches, um_tr, um_dt, row_cost, artifacts.log