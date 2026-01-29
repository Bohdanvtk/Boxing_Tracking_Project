import yaml, random, numpy as np
import tensorflow as tf

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



def make_match_config(cfg: dict) -> MatchConfig:
    alpha = float(_get(cfg, "tracking.matching.alpha", 0.8))
    chi2_gating = float(_get(cfg, "tracking.matching.chi2_gating", 9.21))
    large_cost = float(_get(cfg, "tracking.matching.large_cost", 1e6))
    min_kp_conf = float(_get(cfg, "tracking.matching.min_kp_conf", 0.05))
    greedy_threshold = float(_get(cfg, "tracking.matching.greedy_threshold", 2.8))
    greedy_reset_threshold = float(_get(cfg, "tracking.matching.greedy_reset_threshold", 1))
    keypoint_weights = _get(cfg, "tracking.matching.keypoint_weights", None)
    w_motion = _get(cfg, "tracking.matching.w_motion", 3)
    w_pose = _get(cfg, "tracking.matching.w_pose", 5)
    w_app = _get(cfg, "tracking.matching.w_app", 10)
    pose_core = _get(cfg, "tracking.matching.pose_core", [1, 2, 5, 8, 9, 12] )

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
        w_motion=w_motion,
        greedy_threshold=greedy_threshold,
        greedy_reset_threshold=greedy_reset_threshold,
        w_pose=w_pose,
        w_app=w_app,
        pose_scale_eps=pose_scale_eps,
        save_log=save_log,
        pose_core=pose_core

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

    max_age = int(_get(cfg, "tracking.tracker.max_age", 10))
    min_hits = int(_get(cfg, "tracking.tracker.min_hits", 3))
    min_kp_conf = float(_get(cfg, "tracking.tracker.min_kp_conf", 0.05))
    reset_g_threshold = float(_get(cfg, "tracking.tracker.reset_g_threshold", 0.7))


    return TrackerConfig(
        dt=dt,
        process_var=process_var,
        measure_var=measure_var,
        p0=p0,
        max_age=max_age,
        min_hits=min_hits,
        match=match_cfg,
        min_kp_conf=min_kp_conf,
        reset_g_threshold=reset_g_threshold,
        debug=debug,
        save_log=save_log

    )


def load_tracking_config(path: str):
    cfg = load_cfg(path)
    match_cfg = make_match_config(cfg)
    tracker_cfg = make_tracker_config(cfg, match_cfg)
    return tracker_cfg, match_cfg, cfg


