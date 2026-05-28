from __future__ import annotations

import contextlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

from boxing_project.tracking.inference_utils import init_openpose_from_config, visualize_sequence
from boxing_project.tracking.progress_utils import RichStageProgress
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


@contextlib.contextmanager
def suppress_native_stdout(enabled: bool = True):
    """
    Suppress native stdout noise, for example OpenPose startup messages.

    This suppresses stdout only inside this context block.
    It does not affect the progress hotbar after the block ends.
    """
    if not enabled:
        yield
        return

    stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(stdout_fd)

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(saved_stdout_fd)


@dataclass
class InferRunner:
    """
    One-config inference runner.

    Usage:
        from boxing_project.tracking.infer_runner import InferRunner
        InferRunner("configs/infer_tracks.yaml").run()
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
        verbose = bool(cfg.get("verbose", False))
        restore_mode = bool(cfg.get("restore_mode", False))

        progress = RichStageProgress(
            enabled=bool(cfg.get("progress", {}).get("enabled", True)),
            restore_mode=restore_mode,
        )

        try:
            progress.add("setup", "[0/5] SETUP / INITIALIZATION", total=7)

            # ---------- 1. OpenPose config ----------
            progress.update(
                "setup",
                completed=1,
                description="[0/5] CHECKING OPENPOSE CONFIG",
                force=True,
            )

            if "openpose" not in cfg:
                raise KeyError("infer_tracks.yaml missing key: openpose")

            # ---------- 2. OpenPose ----------
            progress.update(
                "setup",
                completed=2,
                description="[0/5] INITIALIZING OPENPOSE",
                force=True,
            )

            with suppress_native_stdout(enabled=not verbose):
                _, opWrapper = init_openpose_from_config(cfg["openpose"])

            # ---------- 3. Tracking ----------
            progress.update(
                "setup",
                completed=3,
                description="[0/5] BUILDING TRACKER AND MODELS",
                force=True,
            )

            tracking_cfg = cfg.get("tracking", {})
            tracking_cfg_path = _resolve(pr, tracking_cfg.get("config_path"))

            if tracking_cfg_path is None or not tracking_cfg_path.exists():
                raise FileNotFoundError(f"Tracking config not found: {tracking_cfg_path}")

            tracker = MultiObjectTracker(config_path=str(tracking_cfg_path))
            tracking_runtime_cfg = _load_yaml(tracking_cfg_path) or {}

            # ---------- 4. Data / Images ----------
            progress.update(
                "setup",
                completed=4,
                description="[0/5] PREPARING INPUT FRAMES",
                force=True,
            )

            data_cfg = cfg.get("data", {})
            images = load_inference_images(data_cfg, pr)

            save_width = int(data_cfg.get("save_width", 800))
            graph_clustering_params = (
                (tracking_runtime_cfg.get("tracking", {}) or {}).get("graph_clustering", {})
            )

            # ---------- 5. Output directory ----------
            progress.update(
                "setup",
                completed=5,
                description="[0/5] PREPARING OUTPUT DIRECTORY",
                force=True,
            )

            save_dir_raw = data_cfg.get("save_dir", None)
            save_dir = _resolve(pr, save_dir_raw) if save_dir_raw else None

            if save_dir is not None:
                if restore_mode:
                    # IMPORTANT:
                    # Restore mode must keep the output directory because it contains:
                    # - manifests
                    # - checkpoints
                    # - preprocessed parquet files
                    # - local tracking state
                    # - global state
                    progress.message(f"Restore mode enabled: keeping output directory: {save_dir}")
                    save_dir.mkdir(parents=True, exist_ok=True)
                else:
                    # STRONG safety guard:
                    # allow deletion ONLY for .../boxing_tracker/test
                    if (
                        save_dir.exists()
                        and save_dir.name == "test"
                        and save_dir.parent.name == "boxing_tracker"
                    ):
                        progress.message(f"Removing old output directory: {save_dir}")
                        shutil.rmtree(save_dir)
                    elif save_dir.exists():
                        progress.warning(
                            f"Not deleting directory outside safe test path: {save_dir}"
                        )

                    save_dir.mkdir(parents=True, exist_ok=True)

            # ---------- 6. Optional embeddings + shot boundary config ----------
            progress.update(
                "setup",
                completed=6,
                description="[0/5] RESOLVING EMBEDDINGS AND SHOT BOUNDARY",
                force=True,
            )

            app_emb_path = tracking_cfg.get("apperance_embedding_model_path")
            app_emb_path = str(app_emb_path).strip() if app_emb_path is not None else ""
            app_emb_path = app_emb_path if app_emb_path else None

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

            # ---------- 7. Setup complete ----------
            progress.update(
                "setup",
                completed=7,
                description="[0/5] SETUP COMPLETE",
                force=True,
            )

            # ---------- RUN ----------
            visualize_sequence(
                opWrapper=opWrapper,
                tracker=tracker,
                app_emb_path=app_emb_path,
                sb_cfg=sb_cfg,
                images=images,
                save_width=save_width,
                save_dir=save_dir,
                graph_clustering_params=graph_clustering_params,
                pipeline_cfg=cfg,
                progress=progress,
            )

        except Exception:
            try:
                progress.warning("Pipeline failed with an exception")
            except Exception:
                pass
            raise

        finally:
            progress.finish()