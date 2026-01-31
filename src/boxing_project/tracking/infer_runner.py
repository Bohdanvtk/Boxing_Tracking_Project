from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

from boxing_project.tracking.inference_utils import init_openpose_from_config, visualize_sequence
from boxing_project.tracking.tracker import MultiObjectTracker
from boxing_project.tracking.video_utils import load_inference_images


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve(project_root: Path, p: Optional[str | Path]) -> Optional[Path]:
    """Resolve path relative to project_root if it's not absolute."""
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else (project_root / p)


@dataclass
class InferRunner:
    """
    One-config inference runner.

    Usage:
        from boxing_project.tracking.infer_runner import InferRunner
        InferRunner("configs/infer_tracks.yaml").run()
        InferRunner("configs/infer_tracks.yaml").run(video=True)
    """
    infer_cfg_path: str | Path

    def __post_init__(self) -> None:
        self.infer_cfg_path = Path(self.infer_cfg_path)

        if not self.infer_cfg_path.exists():
            raise FileNotFoundError(f"infer_tracks.yaml not found: {self.infer_cfg_path}")

        # infer_cfg expected at: <project_root>/configs/infer_tracks.yaml
        # => project_root = parent of "configs"
        self.project_root = self.infer_cfg_path.parent.parent
        self.cfg = _load_yaml(self.infer_cfg_path)

    def run(self) -> None:
        cfg = self.cfg
        pr = self.project_root

        # ---------- OpenPose ----------
        if "openpose" not in cfg:
            raise KeyError("infer_tracks.yaml missing key: openpose")
        _, opWrapper = init_openpose_from_config(cfg["openpose"])

        # ---------- Tracking ----------
        tracking_cfg = cfg.get("tracking", {})
        tracking_cfg_path = _resolve(pr, tracking_cfg.get("config_path"))
        if tracking_cfg_path is None or not tracking_cfg_path.exists():
            raise FileNotFoundError(f"Tracking config not found: {tracking_cfg_path}")

        tracker = MultiObjectTracker(config_path=str(tracking_cfg_path))

        match_cfg = cfg.get("match", {}) if isinstance(cfg.get("match", {}), dict) else {}
        if match_cfg:
            if "debug_pose_presence" in match_cfg:
                tracker.cfg.match.debug_pose_presence = bool(match_cfg.get("debug_pose_presence"))
            if "debug_motion_centers" in match_cfg:
                tracker.cfg.match.debug_motion_centers = bool(match_cfg.get("debug_motion_centers"))

        # ---------- Data / Images ----------
        data_cfg = cfg.get("data", {})
        images = load_inference_images(data_cfg, pr)

        save_width = int(data_cfg.get("save_width", 800))
        merge_n = int(tracking_cfg.get("num_frames_merge", 40))

        # NEW: artifacts output dir
        save_dir_raw = data_cfg.get("save_dir", None)
        save_dir = _resolve(pr, save_dir_raw) if save_dir_raw else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        # ---------- Optional embeddings ----------
        app_emb_path = tracking_cfg.get("apperance_embedding_model_path")
        app_emb_path = str(app_emb_path).strip() if app_emb_path is not None else ""
        app_emb_path = app_emb_path if app_emb_path else None

        # ---------- Shot boundary (REQUIRED) ----------
        sb_block = cfg.get("shot_boundary", None)
        if not isinstance(sb_block, dict):
            raise KeyError("infer_tracks.yaml missing key: shot_boundary")

        sb_cfg_path = _resolve(pr, sb_block.get("config_path"))
        if sb_cfg_path is None or not sb_cfg_path.exists():
            raise FileNotFoundError(f"Shot boundary config not found: {sb_cfg_path}")

        sb_yaml = _load_yaml(sb_cfg_path)
        sb_cfg = sb_yaml.get("shot_boundary", sb_yaml)

        if not isinstance(sb_cfg, dict) or not sb_cfg:
            raise ValueError(f"Shot boundary config is empty or invalid: {sb_cfg_path}")

        # ---------- RUN ----------
        visualize_sequence(
            opWrapper=opWrapper,
            tracker=tracker,
            app_emb_path=app_emb_path,
            sb_cfg=sb_cfg,
            images=images,
            save_width=save_width,
            merge_n=merge_n,
            save_dir=save_dir,
            show_pose_extended=bool(
                match_cfg.get("debug_pose_presence", False)
                or match_cfg.get("debug_motion_centers", False)
            ),
        )
