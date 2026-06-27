from __future__ import annotations

from typing import Any, Dict, List, Optional

from .track import Detection, Track
from .tracking_debug import (
    append_birth_debug,
    format_birth_debug_lines,
    format_freeze_debug_lines,
    format_removed_tracks_lines,
    format_track_update_debug_lines,
)


def build_used_config_debug(cfg: Any, birth_manager: Any) -> Dict[str, Any]:
    used = {
        "tracking.min_hits": int(cfg.min_hits),
        "tracking.min_hits_sub": int(cfg.min_hits_sub),
        "tracking.max_age": int(cfg.max_age),
        "tracking.max_confirmed_age": int(cfg.max_confirmed_age),
        "tracking.tracker.max_unconfirmed_tracks": int(cfg.max_unconfirmed_tracks),
        "tracking.overlap_log_threshold": float(cfg.overlap_log_threshold),
        "tracking.skeleton_overlap_threshold": float(cfg.skeleton_overlap_threshold),
        "tracking.skeleton_overlap_full_weight": float(cfg.skeleton_overlap_full_weight),
        "tracking.skeleton_overlap_core_weight": float(cfg.skeleton_overlap_core_weight),
        "tracking.skeleton_overlap_conf_threshold": float(cfg.skeleton_overlap_conf_threshold),
        "tracking.skeleton_overlap_thickness": int(cfg.skeleton_overlap_thickness),
        "tracking.adaptive_overlap_center_near": float(cfg.adaptive_overlap_center_near),
        "tracking.adaptive_overlap_center_mid": float(cfg.adaptive_overlap_center_mid),
        "tracking.adaptive_overlap_center_far": float(cfg.adaptive_overlap_center_far),
        "tracking.adaptive_overlap_iou_near": float(cfg.adaptive_overlap_iou_near),
        "tracking.adaptive_overlap_iou_mid": float(cfg.adaptive_overlap_iou_mid),
        "tracking.adaptive_overlap_iou_far": float(cfg.adaptive_overlap_iou_far),
        "tracking.adaptive_overlap_iou_default": float(cfg.adaptive_overlap_iou_default),
        "tracking.overlap_app_freeze_after": int(cfg.overlap_app_freeze_after),
        "tracking.overlap_motion_alpha": float(cfg.overlap_motion_alpha),
    }

    for name, value in vars(cfg.match).items():
        used[f"match.{name}"] = value

    birth_cfg = getattr(birth_manager, "cfg", None)
    if birth_cfg is not None:
        for name, value in vars(birth_cfg).items():
            used[f"birth.{name}"] = value

    return used


def write_match_meta(
    det: Detection,
    trk: Track,
    *,
    update_cost: float,
    row_cost: float,
    d_motion: float,
    d_pose: float,
    d_app: float,
    max_update_cost: float,
) -> None:
    det.meta.update({
        "match_cost": update_cost,
        "match_row_cost": row_cost,
        "match_update_cost": update_cost,
        "match_d_motion": d_motion,
        "match_d_pose": d_pose,
        "match_d_app": d_app,
        "match_update_threshold": max_update_cost,
        "matched_track_id_before_update": int(trk.track_id),
    })


def write_update_gate_meta(det: Detection, disabled_reasons: List[str]) -> None:
    det.meta["track_update_skip_reason"] = ",".join(disabled_reasons) if disabled_reasons else None
    det.meta["track_update_disabled_reasons"] = disabled_reasons


def write_app_decision_meta(det: Detection, app_decision: Dict[str, Any], cfg: Any) -> None:
    det.meta.update({
        "app_update_mode": str(app_decision["mode"]),
        "app_strict_update": app_decision["strict"],
        "app_recovery_candidate": app_decision["recovery_candidate"],
        "app_buffer_upper_eff": app_decision["buffer_upper_eff"],
        "app_buffer_base_upper": app_decision["buffer_base_upper"],
        "app_buffer_hard_upper": app_decision["buffer_hard_upper"],
        "app_stale_frames_before": app_decision["stale_before"],
        "app_stale_frames_after": app_decision["stale_after"],
        "app_buffer_size_before": app_decision["buffer_size_before"],
        "app_buffer_size_after": app_decision["buffer_size_after"],
        "app_buffer_min_size": int(getattr(cfg.match, "app_buffer_min_size", 3)),
        "app_buffer_clear_reason": app_decision["clear_reason"],
        "app_buffer_reject_reason": app_decision["reject_reason"],
        "app_coverage": app_decision["coverage"],
        "app_buffer_motion_ok": app_decision["motion_ok"],
        "app_buffer_pose_ok": app_decision["pose_ok"],
        "app_buffer_coverage_ok": app_decision["coverage_ok"],
        "app_buffer_overlap_ok": app_decision["overlap_ok"],
        "app_buffer_freeze_ok": app_decision["freeze_ok"],
        "app_recovery_batch_update_applied": app_decision["recovery_batch_applied"],
    })


def build_track_update_record(
    *,
    track_idx: int,
    det_idx: int,
    trk: Track,
    det: Detection,
    d_motion: float,
    d_pose: float,
    d_app: float,
    row_cost: float,
    update_cost: float,
    max_update_cost: float,
    max_update_motion: float,
    max_update_pose: float,
    max_update_app: float,
    update_motion: bool,
    update_pose: bool,
    update_app: bool,
) -> Dict[str, Any]:
    meta = det.meta
    return {
        "track_idx": int(track_idx),
        "track_id": int(trk.track_id),
        "det_idx": int(det_idx),
        "hits_before_update": meta.get("track_hits_before_update"),
        "hits_after_update": int(trk.hits),
        "sub_confirmed_before_update": meta.get("track_sub_confirmed_before_update"),
        "sub_confirmed_after_update": bool(trk.sub_confirmed),
        "confirmed_before_update": meta.get("track_confirmed_before_update"),
        "confirmed_after_update": bool(trk.confirmed),
        "d_motion": d_motion,
        "d_pose": d_pose,
        "d_app": d_app,
        "max_update_motion": max_update_motion,
        "max_update_pose": max_update_pose,
        "max_update_app": max_update_app,
        "update_motion": bool(meta.get("track_update_motion_allowed", update_motion)),
        "update_pose": bool(meta.get("track_update_pose_allowed", update_pose)),
        "update_app": bool(meta.get("track_update_app_requested", update_app)),
        "track_match_had_overlap": bool(meta.get("track_match_had_overlap", False)),
        "birth_overlap_bypass": bool(meta.get("birth_overlap_bypass", False)),
        "row_cost": row_cost,
        "update_cost": update_cost,
        "max_update_cost": max_update_cost,
        "track_update_skipped": bool(meta.get("track_update_skipped", False)),
        "track_update_fully_skipped": bool(meta.get("track_update_fully_skipped", False)),
        "track_update_partially_skipped": bool(meta.get("track_update_partially_skipped", False)),
        "track_update_skip_reason": meta.get("track_update_skip_reason"),
        "track_app_update_allowed": bool(meta.get("track_app_update_allowed", False)),
        "track_app_update_block_reason": meta.get("track_app_update_block_reason"),
        "app_update_mode": meta.get("app_update_mode"),
        "app_buffer_upper_eff": meta.get("app_buffer_upper_eff"),
        "app_stale_frames_before": meta.get("app_stale_frames_before"),
        "app_stale_frames_after": meta.get("app_stale_frames_after"),
        "app_buffer_size_before": meta.get("app_buffer_size_before"),
        "app_buffer_size_after": meta.get("app_buffer_size_after"),
        "app_buffer_clear_reason": meta.get("app_buffer_clear_reason"),
        "app_buffer_reject_reason": meta.get("app_buffer_reject_reason"),
        "app_recovery_batch_update_applied": meta.get("app_recovery_batch_update_applied"),
        "max_overlap_iou": meta.get("max_overlap_iou"),
        "max_overlap_det_idx": meta.get("max_overlap_det_idx"),
        "min_center_dist_norm": meta.get("min_center_dist_norm"),
        "center_dist_norm_det_idx": meta.get("center_dist_norm_det_idx"),
        "active_overlap_threshold": meta.get("active_overlap_threshold"),
        "adaptive_overlap_zone": meta.get("adaptive_overlap_zone"),
        "adaptive_overlap_enabled": meta.get("adaptive_overlap_enabled"),
        "adaptive_overlap_reason": meta.get("adaptive_overlap_reason"),
        "current_track_stable_for_overlap": meta.get("current_track_stable_for_overlap"),
        "overlap_has_stable_track": meta.get("overlap_has_stable_track"),
        "track_has_risky_overlap": meta.get("track_has_risky_overlap"),
        "risky_overlap_count": meta.get("risky_overlap_count"),
        "risky_overlap_det_indices": meta.get("risky_overlap_det_indices"),
        "max_risky_overlap_iou": meta.get("max_risky_overlap_iou"),
        "raw_max_overlap_iou": meta.get("raw_max_overlap_iou"),
        "overlap_motion_weak_update": meta.get("overlap_motion_weak_update"),
        "overlap_motion_alpha": meta.get("overlap_motion_alpha"),
        "overlap_motion_update_center": meta.get("overlap_motion_update_center"),
    }


def attach_track_update_debug(log: Any, used_config: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    log.meta["used_config"] = used_config
    log.meta["track_update_debug"] = records
    lines = format_track_update_debug_lines(records)
    if lines and hasattr(log, "buffer") and isinstance(log.buffer, list):
        log.buffer.extend(["", *lines])


def attach_birth_debug(log: Any, birth_debug: List[Dict[str, Any]]) -> None:
    log.meta["birth_debug"] = birth_debug
    append_birth_debug(birth_debug)
    if hasattr(log, "buffer") and isinstance(log.buffer, list):
        log.buffer.extend(["", *format_birth_debug_lines(birth_debug)])


def build_freeze_debug(tracks: List[Track]) -> List[Dict[str, Any]]:
    return [
        {
            "track_idx": int(idx),
            "track_id": int(track.track_id),
            "freeze_active": bool(track.is_frozen()),
            "freeze_frames_left": int(getattr(track, "freeze_frames_left", 0)),
            "freeze_sources": {
                int(source_id): int(frames_left)
                for source_id, frames_left in getattr(track, "freeze_sources", {}).items()
            },
            "overlap_group_ids": sorted(int(x) for x in getattr(track, "overlap_group_ids", set())),
        }
        for idx, track in enumerate(tracks)
    ]


def attach_freeze_debug(log: Any, tracks: List[Track]) -> None:
    freeze_debug = build_freeze_debug(tracks)
    log.meta["freeze_debug"] = freeze_debug
    lines = format_freeze_debug_lines(freeze_debug)
    if lines and hasattr(log, "buffer") and isinstance(log.buffer, list):
        log.buffer.extend(["", *lines])


def attach_removed_tracks_debug(
    log: Any,
    *,
    pruned: List[Dict[str, Any]],
    dead: List[Dict[str, Any]],
) -> None:
    log.meta["removed_dead_tracks"] = dead
    lines = format_removed_tracks_lines(pruned=pruned, dead=dead)
    if lines and hasattr(log, "buffer") and isinstance(log.buffer, list):
        log.buffer.extend(["", *lines])


def build_active_tracks_summary(
    tracks: List[Track],
    *,
    max_age: int,
    max_confirmed_age: int,
) -> List[Dict[str, Any]]:
    return [
        {
            "track_id": t.track_id,
            "sub_confirmed": t.sub_confirmed,
            "hits": t.hits,
            "confirmed": t.confirmed,
            "age": t.age,
            "time_since_update": t.time_since_update,
            "state": t.state.tolist(),
            "pos": t.pos(),
            "overlap_group_ids": sorted(int(x) for x in getattr(t, "overlap_group_ids", set())),
            "freeze_sources": {
                int(source_id): int(frames_left)
                for source_id, frames_left in getattr(t, "freeze_sources", {}).items()
            },
            "freeze_frames_left": int(getattr(t, "freeze_frames_left", 0)),
            "app_emb_history_len": int(len(getattr(t, "app_emb_history", []) or [])),
        }
        for t in tracks
        if not t.is_dead(max_age, max_confirmed_age)
    ]
