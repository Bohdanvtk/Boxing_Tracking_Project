from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .pairs_dataset import PairRecord, PairsDataset
from .losses import ContrastiveLoss
from .siamese_model import SiameseOSNet


@dataclass
class StageRule:
    lr_stage: float
    patience: int
    restore_best_weights: bool


def get_stage_rule(root: Path, base_lr: float, early_stop_patience: int) -> StageRule:
    return StageRule(
        lr_stage=base_lr,
        patience=early_stop_patience,
        restore_best_weights=True,
    )


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs_per_stage: int = 10
    lr: float = 1e-4
    margin: float = 1.0
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42

    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    scheduler_min_lr: float = 1e-6

    early_stop_patience: int = 12


class StageTrainer:
    def __init__(
        self,
        model: SiameseOSNet,
        transform,
        cfg: TrainConfig,
        out_dir: Path,
    ):
        self.model = model
        self.transform = transform
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if cfg.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = cfg.device

        self.model.to(self.device)
        self.loss_fn = ContrastiveLoss(margin=cfg.margin)

    def _make_loader(self, pairs: List[PairRecord], shuffle: bool) -> DataLoader:
        ds = PairsDataset(pairs=pairs, transform=self.transform)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=(self.device == "cuda"),
            drop_last=False,
        )

    @staticmethod
    def _weighted_reduce(per_sample_loss: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
        per_sample_loss = per_sample_loss.view(-1)
        sample_weight = sample_weight.view(-1)
        return (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)

    def _run_epoch(self, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer]) -> float:
        train_mode = optimizer is not None
        self.model.train(train_mode)

        total_weighted_loss = 0.0
        total_weight = 0.0

        it = tqdm(loader, desc=("train" if train_mode else "val"), leave=False)
        for a, b, y, w in it:
            a = a.to(self.device, non_blocking=True)
            b = b.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            w = w.to(self.device, non_blocking=True)

            dist = self.model.forward_pair(a, b)
            per_sample_loss = self.loss_fn(dist, y)
            loss = self._weighted_reduce(per_sample_loss, w)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_weighted_loss = float((per_sample_loss.view(-1) * w.view(-1)).sum().detach().cpu())
            batch_weight_sum = float(w.sum().detach().cpu())

            total_weighted_loss += batch_weighted_loss
            total_weight += batch_weight_sum

            it.set_postfix(loss=float(loss.detach().cpu()))

        return total_weighted_loss / max(total_weight, 1e-12)

    def train_stages(
        self,
        train_roots: List[Path],
        train_pairs_by_root: List[List[PairRecord]],
        val_pairs: List[PairRecord],
        start_epoch: int = 1,
    ) -> Path:
        assert len(train_roots) == len(train_pairs_by_root)

        val_loader = self._make_loader(val_pairs, shuffle=False)

        history = []
        best_final_path = None

        for stage_idx, (root, stage_pairs) in enumerate(zip(train_roots, train_pairs_by_root), start=1):
            rule = get_stage_rule(root, self.cfg.lr, self.cfg.early_stop_patience)
            print(f"\n=== Stage {stage_idx}/{len(train_roots)}: {root}")
            print(f"lr_stage={rule.lr_stage} patience={rule.patience} restore_best={rule.restore_best_weights}")
            print(f"num_pairs={len(stage_pairs)}")

            train_loader = self._make_loader(stage_pairs, shuffle=True)

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable_params:
                raise RuntimeError("No trainable parameters found.")

            optimizer = torch.optim.Adam(trainable_params, lr=rule.lr_stage)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.cfg.scheduler_factor,
                patience=self.cfg.scheduler_patience,
                min_lr=self.cfg.scheduler_min_lr,
            )

            stage_dir = self.out_dir / f"stage_{stage_idx:02d}"
            stage_dir.mkdir(parents=True, exist_ok=True)

            # Resume from best.pth when it already exists in the stage directory.
            best_path = stage_dir / "best.pth"
            if best_path.exists():
                best_state = torch.load(best_path, map_location="cpu")
                print(f"[continue] Existing best checkpoint found: {best_path}")
            else:
                best_state = None

            best_val = float("inf")
            no_improve = 0

            if start_epoch > self.cfg.epochs_per_stage:
                print(
                    f"[continue] Nothing to do: start_epoch={start_epoch} "
                    f"> epochs_per_stage={self.cfg.epochs_per_stage}"
                )
                continue

            for epoch in range(start_epoch, self.cfg.epochs_per_stage + 1):
                train_loss = self._run_epoch(train_loader, optimizer)
                val_loss = self._run_epoch(val_loader, optimizer=None)

                current_lr = optimizer.param_groups[0]["lr"]

                print(
                    f"[stage {stage_idx}] epoch {epoch:03d}: "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={current_lr:.8f}"
                )

                history.append(
                    {
                        "stage": stage_idx,
                        "root": str(root),
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "lr": current_lr,
                    }
                )

                epoch_path = stage_dir / f"epoch_{epoch:03d}_valloss_{val_loss:.6f}.pth"
                torch.save(self.model.state_dict(), epoch_path)

                # Update best when the current epoch is the best one seen in this run.
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improve = 0

                    torch.save(best_state, best_path)
                    best_final_path = best_path
                else:
                    no_improve += 1

                scheduler.step(val_loss)

                new_lr = optimizer.param_groups[0]["lr"]
                if new_lr != current_lr:
                    print(f"[stage {stage_idx}] lr reduced: {current_lr:.8f} -> {new_lr:.8f}")

                # Save last.pth as a convenient latest checkpoint.
                last_path = stage_dir / "last.pth"
                torch.save(self.model.state_dict(), last_path)

                if no_improve >= rule.patience:
                    print(f"[stage {stage_idx}] early stopping (patience={rule.patience})")
                    break

            if rule.restore_best_weights and best_state is not None:
                self.model.load_state_dict(best_state)

        final_path = self.out_dir / "final.pth"
        torch.save(self.model.state_dict(), final_path)

        history_path = self.out_dir / "history.json"
        old_history = []
        if history_path.exists():
            try:
                old_history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                old_history = []

        all_history = old_history + history
        history_path.write_text(
            json.dumps(all_history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"\nSaved final checkpoint: {final_path}")
        if best_final_path:
            print(f"Last saved stage best: {best_final_path}")
        return final_path