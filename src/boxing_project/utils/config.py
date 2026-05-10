import yaml, random, numpy as np
import tensorflow as tf
from pathlib import Path

def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def _get(d: dict, path: str, default=None):

    cur = d
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


from boxing_project.tracking.matcher import MatchConfig
from boxing_project.tracking.tracker import TrackerConfig
from boxing_project.tracking.birth_manager import BirthConfig



def make_match_config(cfg: dict) -> MatchConfig:
    alpha = float(_get(cfg, "tracking.matching.alpha", 0.8))
    chi2_gating = float(_get(cfg, "tracking.matching.chi2_gating", 9.21))
    large_cost = float(_get(cfg, "tracking.matching.large_cost", 1e6))
    min_kp_conf = float(_get(cfg, "tracking.matching.min_kp_conf", 0.05))
    greedy_threshold = float(_get(cfg, "tracking.matching.greedy_threshold", 2.8))
    max_update_cost = float(_get(cfg, "tracking.matching.max_update_cost", 1.2))
    max_update_motion = float(_get(cfg, "tracking.matching.max_update_motion", 0.08))
    max_update_pose = float(_get(cfg, "tracking.matching.max_update_pose", 0.30))
    max_update_app = float(_get(cfg, "tracking.matching.max_update_app", 0.12))
    motion_threshold = float(_get(cfg, "tracking.matching.motion_threshold", 1.0))
    pose_threshold = float(_get(cfg, "tracking.matching.pose_threshold", 1.0))
    appearance_threshold = float(_get(cfg, "tracking.matching.appearance_threshold", 1.0))
    emb_ema_alpha = float(_get(cfg, "tracking.matching.emb_ema_alpha", 0.9))
    keypoint_weights = _get(cfg, "tracking.matching.keypoint_weights", None)
    w_motion = _get(cfg, "tracking.matching.w_motion", 3)
    w_pose = _get(cfg, "tracking.matching.w_pose", 5)
    w_app = _get(cfg, "tracking.matching.w_app", 10)
    k_motion = _get(cfg, "tracking.matching.k_motion", 3)
    k_pose = _get(cfg, "tracking.matching.k_pose", 3)
    k_app = _get(cfg, "tracking.matching.k_app", 4)
    miss_relax_full_after = int(_get(cfg, "tracking.matching.miss_relax_full_after", 20))
    miss_relax_strength = float(_get(cfg, "tracking.matching.miss_relax_strength", 2.0))
    min_core_kps = int(_get(cfg, "tracking.matching.min_core_kps", 8))
    pose_core = _get(cfg, "tracking.matching.pose_core", [1, 2, 5, 8, 9, 12] )
    pose_center = _get(cfg, "tracking.matching.pose_center", [8, 1, 9, 12, 5, 2])

    save_log = bool(_get(cfg, "tracking.save_log", True))

    pose_scale_eps = float(_get(cfg, "tracking.matching.pose_scale_eps", 1e-6))


    if keypoint_weights is not None and not isinstance(keypoint_weights, (list, tuple)):
        raise ValueError("keypoint_weights має бути списком чисел або None")


    return MatchConfig(
        alpha=alpha,
        chi2_gating=chi2_gating,
        large_cost=large_cost,
        min_kp_conf=min_kp_conf,
        keypoint_weights=keypoint_weights,

        greedy_threshold=greedy_threshold,
        max_update_cost=max_update_cost,
        max_update_motion=max_update_motion,
        max_update_pose=max_update_pose,
        max_update_app=max_update_app,
        motion_threshold=motion_threshold,
        pose_threshold=pose_threshold,
        appearance_threshold=appearance_threshold,
        emb_ema_alpha=emb_ema_alpha,
        w_motion=w_motion,
        w_pose=w_pose,
        w_app=w_app,
        k_motion=k_motion,
        k_pose=k_pose,
        k_app=k_app,
        miss_relax_full_after=miss_relax_full_after,
        miss_relax_strength=miss_relax_strength,
        min_core_kps=min_core_kps,
        pose_scale_eps=pose_scale_eps,
        save_log=save_log,
        pose_core=pose_core,
        pose_center=pose_center
    )
def make_tracker_config(cfg: dict, match_cfg: MatchConfig) -> TrackerConfig:
    fps = _get(cfg, "tracking.fps", None)

    if fps is not None:
        dt = 1.0 / float(fps)
    else:
        dt = float(_get(cfg, "tracking.kalman.dt", 1.0 / 60.0))

    process_var = float(_get(cfg, "tracking.kalman.process_var", 3.0))
    measure_var = float(_get(cfg, "tracking.kalman.measure_var", 9.0))
    p0 = float(_get(cfg, "tracking.kalman.p0", 1000.0))

    debug = bool(_get(cfg, "tracking.debug", False))
    save_log = bool(_get(cfg, "tracking.save_log", True))

    overlap_log_threshold = float(_get(cfg, "tracking.overlap_log_threshold", 0.10))
    # Keep defaults here so older YAMLs still load.
    adaptive_overlap_center_near = float(_get(cfg, "tracking.adaptive_overlap_center_near", 0.55))
    adaptive_overlap_center_mid = float(_get(cfg, "tracking.adaptive_overlap_center_mid", 0.85))
    adaptive_overlap_center_far = float(_get(cfg, "tracking.adaptive_overlap_center_far", 1.20))
    adaptive_overlap_iou_near = float(_get(cfg, "tracking.adaptive_overlap_iou_near", 0.03))
    adaptive_overlap_iou_mid = float(_get(cfg, "tracking.adaptive_overlap_iou_mid", 0.06))
    adaptive_overlap_iou_far = float(_get(cfg, "tracking.adaptive_overlap_iou_far", 0.08))
    adaptive_overlap_iou_default = float(_get(cfg, "tracking.adaptive_overlap_iou_default", 0.12))
    overlap_app_freeze_after = int(_get(cfg, "tracking.overlap_app_freeze_after", 5))

    max_age = int(_get(cfg, "tracking.tracker.max_age", 10))
    max_confirmed_age = int(_get(cfg, "tracking.tracker.max_confirmed_age", 40))
    min_hits = int(_get(cfg, "tracking.tracker.min_hits", 3))
    min_hits_sub = int(_get(cfg, "tracking.tracker.min_hits_sub", max(1, min_hits - 1)))
    min_kp_conf = float(_get(cfg, "tracking.tracker.min_kp_conf", 0.05))
    reset_g_threshold = float(_get(cfg, "tracking.tracker.reset_g_threshold", 0.7))

    return TrackerConfig(
        dt=dt,
        process_var=process_var,
        measure_var=measure_var,
        p0=p0,
        max_age=max_age,
        max_confirmed_age=max_confirmed_age,
        min_hits=min_hits,
        min_hits_sub=min_hits_sub,
        match=match_cfg,
        min_kp_conf=min_kp_conf,
        reset_g_threshold=reset_g_threshold,
        debug=debug,
        save_log=save_log,
        overlap_log_threshold=overlap_log_threshold,
        adaptive_overlap_center_near=adaptive_overlap_center_near,
        adaptive_overlap_center_mid=adaptive_overlap_center_mid,
        adaptive_overlap_center_far=adaptive_overlap_center_far,
        adaptive_overlap_iou_near=adaptive_overlap_iou_near,
        adaptive_overlap_iou_mid=adaptive_overlap_iou_mid,
        adaptive_overlap_iou_far=adaptive_overlap_iou_far,
        adaptive_overlap_iou_default=adaptive_overlap_iou_default,
        overlap_app_freeze_after=overlap_app_freeze_after,
    )

def load_tracking_config(path: str):
    cfg = load_cfg(path)
    match_cfg = make_match_config(cfg)
    tracker_cfg = make_tracker_config(cfg, match_cfg)
    return tracker_cfg, match_cfg, cfg


def load_birth_config(path: str) -> BirthConfig:
    cfg = load_cfg(path) if Path(path).exists() else {}
    b = _get(cfg, "birth_manager", {}) or {}
    return BirthConfig(
        chi2_gating=float(b.get("chi2_gating", 9.4877)),
        max_pending_age=int(b.get("max_pending_age", 4)),
        max_pending_misses=int(b.get("max_pending_misses", 2)),
        very_close_threshold=float(b.get("very_close_threshold", 0.04)),
        near_threshold=float(b.get("near_threshold", 0.15)),
        pending_motion_threshold=float(b.get("pending_motion_threshold", 0.20)),
        normal_confirm_hits=int(b.get("normal_confirm_hits", 2)),
        near_confirm_hits=int(b.get("near_confirm_hits", 4)),
        emb_ema_alpha=float(b.get("emb_ema_alpha", 0.9)),
        min_kp_conf=float(b.get("min_kp_conf", 0.05)),
        min_core_kps=int(b.get("min_core_kps", 3)),
        pose_missing_penalty=float(b.get("pose_missing_penalty", 0.05)),
        pose_bad_penalty=float(b.get("pose_bad_penalty", 0.18)),
        app_missing_penalty=float(b.get("app_missing_penalty", 0.03)),
        app_bad_penalty=float(b.get("app_bad_penalty", 0.12)),
        app_bad_threshold=float(b.get("app_bad_threshold", 0.35)),
        near_existing_penalty=float(b.get("near_existing_penalty", 0.03)),
    )

