from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WEIGHT_RE = re.compile(r"__w_(\d+)$")
ID_RE = re.compile(r"^(\d+)")


def parse_weight_from_name(path: Path) -> float:
    """
    000123__neg_hard__w_280.jpg -> 2.8
    000124__pos_easy__w_100.jpg -> 1.0
    """
    stem = path.stem
    m = WEIGHT_RE.search(stem)
    if not m:
        return 1.0
    return int(m.group(1)) / 100.0


def parse_pair_id_from_name(path: Path) -> str:
    """
    000123__neg_hard__w_280.jpg -> "000123"
    000123.txt -> "000123"
    """
    stem = path.stem
    m = ID_RE.match(stem)
    if not m:
        raise ValueError(f"Cannot parse pair id from filename: {path.name}")
    return m.group(1)


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    a_path: Path
    b_path: Path
    label: int
    sample_weight: float


def _index_dir_by_pair_id(folder: Path) -> dict[str, Path]:
    items = [p for p in folder.iterdir() if p.is_file()]
    mapping: dict[str, Path] = {}
    for p in items:
        pid = parse_pair_id_from_name(p)
        if pid in mapping:
            raise RuntimeError(f"Duplicate pair id={pid} in {folder}")
        mapping[pid] = p
    return mapping


def load_pairs_from_root(root: Path) -> List[PairRecord]:
    """
    Expected dataset layout:
      root/
        A/
        B/
        Label/

    File format:
      A/<id>__...__w_XXX.jpg
      B/<id>__...__w_XXX.jpg
      Label/<id>.txt   (0 or 1)
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    a_dir = root / "A"
    b_dir = root / "B"
    label_dir = root / "Label"

    if not a_dir.exists() or not b_dir.exists() or not label_dir.exists():
        raise RuntimeError(
            f"Expected dataset structure:\n"
            f"  {root}/A\n"
            f"  {root}/B\n"
            f"  {root}/Label"
        )

    a_files = sorted([p for p in a_dir.iterdir() if is_image_file(p)])
    b_files = sorted([p for p in b_dir.iterdir() if is_image_file(p)])
    label_files = sorted([p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])

    if not a_files:
        raise RuntimeError(f"No images found in {a_dir}")
    if not b_files:
        raise RuntimeError(f"No images found in {b_dir}")
    if not label_files:
        raise RuntimeError(f"No label txt files found in {label_dir}")

    a_map = _index_dir_by_pair_id(a_dir)
    b_map = _index_dir_by_pair_id(b_dir)
    y_map = _index_dir_by_pair_id(label_dir)

    common_ids = sorted(set(a_map.keys()) & set(b_map.keys()) & set(y_map.keys()))
    if not common_ids:
        raise RuntimeError(
            f"No matching pair ids across A/B/Label in {root}"
        )

    missing_in_b = sorted(set(a_map.keys()) - set(b_map.keys()))
    missing_in_y = sorted(set(a_map.keys()) - set(y_map.keys()))
    if missing_in_b[:5] or missing_in_y[:5]:
        raise RuntimeError(
            f"Dataset ids are inconsistent.\n"
            f"Missing in B (first 5): {missing_in_b[:5]}\n"
            f"Missing in Label (first 5): {missing_in_y[:5]}"
        )

    records: List[PairRecord] = []
    for pid in common_ids:
        a_path = a_map[pid]
        b_path = b_map[pid]
        y_path = y_map[pid]

        label_text = y_path.read_text(encoding="utf-8").strip()
        try:
            label = int(float(label_text))
        except Exception as e:
            raise ValueError(f"Invalid label in {y_path}: {label_text}") from e

        if label not in (0, 1):
            raise ValueError(f"Label must be 0 or 1, got {label} in {y_path}")

        w_a = parse_weight_from_name(a_path)
        w_b = parse_weight_from_name(b_path)

        if abs(w_a - w_b) > 1e-8:
            raise RuntimeError(
                f"Weight mismatch for pair_id={pid}: "
                f"A={a_path.name} -> {w_a}, "
                f"B={b_path.name} -> {w_b}"
            )

        records.append(
            PairRecord(
                pair_id=pid,
                a_path=a_path,
                b_path=b_path,
                label=label,
                sample_weight=w_a,
            )
        )

    return records


class PairsDataset(Dataset):
    def __init__(self, pairs: Sequence[PairRecord], transform: Callable):
        if not pairs:
            raise ValueError("PairsDataset got empty pairs list.")
        self.pairs = list(pairs)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        rec = self.pairs[idx]

        a_img = Image.open(rec.a_path).convert("RGB")
        b_img = Image.open(rec.b_path).convert("RGB")

        a = self.transform(a_img)
        b = self.transform(b_img)

        y = torch.tensor([float(rec.label)], dtype=torch.float32)
        w = torch.tensor([float(rec.sample_weight)], dtype=torch.float32)

        return a, b, y, w