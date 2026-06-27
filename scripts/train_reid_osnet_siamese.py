"""
Train the ReID appearance model used by the boxing tracker.

This script fine-tunes an OSNet-style Siamese model on boxer image pairs.
The trained model is later exported to ONNX and used during tracking inference
to compare visual identity between detections and existing tracks.
"""


from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from boxing_project.reid_training.preprocessing import build_osnet_transform
from boxing_project.reid_training.pairs_dataset import load_pairs_from_root
from boxing_project.reid_training.siamese_model import SiameseOSNet, SiameseConfig
from boxing_project.reid_training.stage_trainer import StageTrainer, TrainConfig

device = "cuda" if torch.cuda.is_available() else "cpu"


EPOCH_FILE_RE = re.compile(r"^epoch_(\d+)_")
PHASE_RE = re.compile(r"stage[_-]?(\d+)", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser("Train OSNet Siamese with lightweight resume-from-weights")
    p.add_argument("--train_roots", nargs="+", required=True)
    p.add_argument("--val_root", required=True)

    p.add_argument("--resume_weights", type=str, default=None)
    p.add_argument("--model_name", type=str, default="osnet_x0_5")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs_per_stage", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--scheduler_factor", type=float, default=0.5)
    p.add_argument("--scheduler_patience", type=int, default=2)
    p.add_argument("--scheduler_min_lr", type=float, default=1e-6)
    p.add_argument("--early_stop_patience", type=int, default=12)

    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--allow_unfreeze_last_stage", action="store_true")

    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def infer_phase_from_run_dir_name(run_dir_name: str) -> int:
    """
    Infer the stage number from the run directory name:
      runs_stage1_model_...
      runs_stage2_model_...
    If no stage is found, return 1.
    """
    m = PHASE_RE.search(run_dir_name)
    if not m:
        return 1
    return int(m.group(1))


def infer_completed_epochs(stage_dir: Path) -> int:
    """
    Use the maximum epoch index from files named epoch_XXX_..., not the raw file count.
    """
    max_epoch = 0
    for p in stage_dir.iterdir():
        if not p.is_file():
            continue
        m = EPOCH_FILE_RE.match(p.name)
        if m:
            max_epoch = max(max_epoch, int(m.group(1)))
    return max_epoch


def main():
    args = parse_args()

    train_roots = [Path(r) for r in args.train_roots]
    val_root = Path(args.val_root)

    # If resume_weights is set, continue inside the parent run directory.
    if args.resume_weights is not None:
        ckpt_path = Path(args.resume_weights)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume weights not found: {ckpt_path}")

        stage_dir = ckpt_path.parent            # .../stage_01
        run_dir = stage_dir.parent              # .../runs_stage2_...
        args.out_dir = str(run_dir)

        completed_epochs = infer_completed_epochs(stage_dir)
        start_epoch = completed_epochs + 1
        inferred_phase = infer_phase_from_run_dir_name(run_dir.name)

        print("========== RESUME DEBUG ==========")
        print("resume_weights:", ckpt_path)
        print("run_dir:", run_dir)
        print("stage_dir:", stage_dir)
        print("inferred_phase:", inferred_phase)
        print("completed_epochs:", completed_epochs)
        print("next_epoch:", start_epoch)
        print("==================================")
    else:
        ckpt_path = None
        completed_epochs = 0
        start_epoch = 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = build_osnet_transform(height=256, width=128)

    val_pairs = load_pairs_from_root(val_root)
    print(f"val_pairs={len(val_pairs)} from {val_root}")

    train_pairs_by_root = []
    for r in train_roots:
        pairs = load_pairs_from_root(r)
        print(f"train_pairs={len(pairs)} from {r}")
        train_pairs_by_root.append(pairs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg = SiameseConfig(
        model_name=args.model_name,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
        allow_unfreeze_last_stage=args.allow_unfreeze_last_stage,
    )
    model = SiameseOSNet(model_cfg, device=device)

    print("========== MODEL DEBUG ==========")
    print("Model name:", model_cfg.model_name)
    print("Encoder:", model.encoder)


    print("Params:", sum(p.numel() for p in model.encoder.parameters()) / 1e6, "M")
    print("=================================")

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=True)
        else:
            model.load_state_dict(state, strict=True)
        print(f"Loaded resume weights: {ckpt_path}")

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        epochs_per_stage=args.epochs_per_stage,
        lr=args.lr,
        margin=args.margin,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        scheduler_min_lr=args.scheduler_min_lr,
        early_stop_patience=args.early_stop_patience,
    )

    trainer = StageTrainer(
        model=model,
        transform=transform,
        cfg=train_cfg,
        out_dir=out_dir,
    )

    final_path = trainer.train_stages(
        train_roots=train_roots,
        train_pairs_by_root=train_pairs_by_root,
        val_pairs=val_pairs,
        start_epoch=start_epoch,
    )

    print(f"Done. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()