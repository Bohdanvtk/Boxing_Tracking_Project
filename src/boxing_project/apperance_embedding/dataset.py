from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf


@dataclass
class FolderPairsConfig:
    root_dir: Path
    image_size: Tuple[int, int] = (128, 128)  # (H, W)
    seed: int = 42


class CropPairsFolder:
    """
    root_dir/
      crops_a/000123.jpg
      crops_b/000123.jpg
      labels/000123.txt

    Output:
      ((img_a, img_b), y)
      img_*: uint8 RGB (H,W,3)  (без нормалізації)
      y: float32 0/1
    """
    def __init__(self, cfg: FolderPairsConfig, ids: List[str]):
        self.cfg = cfg
        self.ids = ids

        self.crops_a = cfg.root_dir / "crops_a"
        self.crops_b = cfg.root_dir / "crops_b"
        self.labels = cfg.root_dir / "labels"

    @classmethod
    def from_folder(cls, cfg: FolderPairsConfig) -> "CropPairsFolder":
        crops_a = cfg.root_dir / "crops_a"
        crops_b = cfg.root_dir / "crops_b"
        labels = cfg.root_dir / "labels"

        if not crops_a.exists():
            raise FileNotFoundError(f"Missing folder: {crops_a}")
        if not crops_b.exists():
            raise FileNotFoundError(f"Missing folder: {crops_b}")
        if not labels.exists():
            raise FileNotFoundError(f"Missing folder: {labels}")

        ids = sorted([p.stem for p in crops_a.glob("*.jpg")])
        if not ids:
            raise RuntimeError(f"No jpg files in {crops_a}")

        ok = [
            sid for sid in ids
            if (crops_b / f"{sid}.jpg").exists() and (labels / f"{sid}.txt").exists()
        ]
        if not ok:
            raise RuntimeError("No valid pairs found. Check crops_b/labels filenames match crops_a.")

        return cls(cfg, ok)

    def __len__(self) -> int:
        return len(self.ids)

    def as_tf_dataset(self, batch_size: int, shuffle: bool) -> tf.data.Dataset:
        ids = tf.constant(self.ids)

        base_a = tf.constant(str(self.crops_a))
        base_b = tf.constant(str(self.crops_b))
        base_l = tf.constant(str(self.labels))

        def make_paths(sid: tf.Tensor):
            pa = tf.strings.join([base_a, "/", sid, ".jpg"])
            pb = tf.strings.join([base_b, "/", sid, ".jpg"])
            pl = tf.strings.join([base_l, "/", sid, ".txt"])
            return pa, pb, pl

        def load_img(path: tf.Tensor) -> tf.Tensor:
            img_bytes = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img_bytes, channels=3)  # RGB uint8
            img = tf.image.resize(img, self.cfg.image_size, method="bilinear")
            img = tf.cast(img, tf.uint8)
            return img

        def load_example(sid: tf.Tensor):
            pa, pb, pl = make_paths(sid)

            y_txt = tf.io.read_file(pl)
            y = tf.strings.to_number(tf.strings.strip(y_txt), out_type=tf.int32)
            y = tf.cast(y, tf.float32)

            img_a = load_img(pa)
            img_b = load_img(pb)

            return (img_a, img_b), y

        ds = tf.data.Dataset.from_tensor_slices(ids)

        if shuffle:
            ds = ds.shuffle(
                buffer_size=min(len(self.ids), 2000),
                seed=self.cfg.seed,
                reshuffle_each_iteration=True,
            )

        ds = ds.map(load_example, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
