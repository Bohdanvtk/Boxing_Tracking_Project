# scripts/infer_tracks.py


# (якщо є shebang або коментарі — вони можуть бути вище)

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))



import yaml
from pathlib import Path
import sys

from src.boxing_project.tracking.inference_utils import (
    init_openpose_from_config,
    visualize_sequence,
)
from src.boxing_project.tracking.tracker import MultiObjectTracker


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# !!! do NOT import src.boxing_project.utils.config here !!!

def load_config(path: Path):
    """Load YAML file into a Python dict, without any heavy side-effects."""
    with open(path, "r") as f:
        return yaml.safe_load(f)



def get_project_root() -> Path:
    """
    Resolve project root as the parent of the 'scripts' directory.
    This file is .../Boxing_Project/scripts/infer_tracks.py
    so project root is parents[1].
    """
    return Path(__file__).resolve().parents[1]


def main():
    project_root = get_project_root()

    # --- config path resolved from project root ---
    cfg_path = project_root / "configs" / "infer_tracks.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = load_config(cfg_path)

    # --- OpenPose ---
    _, opWrapper = init_openpose_from_config(cfg["openpose"])

    # --- tracker ---
    tracking_cfg = cfg["tracking"]
    tracking_cfg_path = tracking_cfg["config_path"]

    # if config_path is relative → resolve from project root
    tracking_cfg_path = (
        project_root / tracking_cfg_path
        if not Path(tracking_cfg_path).is_absolute()
        else Path(tracking_cfg_path)
    )

    if not tracking_cfg_path.exists():
        raise FileNotFoundError(f"Tracking config not found: {tracking_cfg_path}")

    tracker = MultiObjectTracker(config_path=str(tracking_cfg_path))

    # --- images ---
    data_cfg = cfg["data"]
    image_dir = Path(data_cfg["image_dir"])
    if not image_dir.is_absolute():
        image_dir = project_root / image_dir

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    images = sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


    if not images:
        raise RuntimeError(f"No images found in directory: {image_dir}")


    pose_emb_path = tracking_cfg['pose_embedding_model_path']
    app_emb_path = tracking_cfg['apperance_embedding_model_path']

    # --- run inference loop ---
    visualize_sequence(
        opWrapper=opWrapper,
        tracker=tracker,
        pose_emb_path=pose_emb_path,
        app_emb_path=app_emb_path,
        images=images,
        save_width=data_cfg["save_width"],
        merge_n=tracking_cfg["num_frames_merge"],

    )


if __name__ == "__main__":
    main()
