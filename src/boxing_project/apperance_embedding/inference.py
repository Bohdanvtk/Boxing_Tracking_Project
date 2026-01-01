from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None

from .preprocessing import preprocess_crops_np


@dataclass(frozen=True)
class AppearanceEmbedConfig:
    """Конфіг для AppearanceEmbedder (шлях до моделі + препроцес)."""
    model_path: str
    to_rgb: bool = True
    l2_normalize: bool = True


class AppearanceEmbedder:
    """Класовий інференс appearance-embedding: вантажимо CNN 1 раз, далі embed() для кожного bbox."""

    def __init__(self, cfg: AppearanceEmbedConfig):
        """Завантажує appearance encoder з cfg.model_path (artifacts) та тримає його в памʼяті."""
        if tf is None:
            raise RuntimeError("TensorFlow не доступний. Перевір встановлення tensorflow у venv.")
        self.cfg = cfg
        self.model = tf.keras.models.load_model(str(Path(cfg.model_path)), compile=False)

    def embed(self, frame_bgr: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Рахує appearance embedding для одного bbox.

        Args:
            frame_bgr: кадр OpenCV (H,W,3) BGR.
            bbox_xyxy: (x1,y1,x2,y2) int.

        Returns:
            np.ndarray (D,) — embedding (опційно L2-normalized).
        """
        x1, y1, x2, y2 = bbox_xyxy
        h, w = frame_bgr.shape[:2]

        x1 = int(max(0, min(w - 1, x1)))
        x2 = int(max(0, min(w, x2)))
        y1 = int(max(0, min(h - 1, y1)))
        y2 = int(max(0, min(h, y2)))

        if x2 <= x1 or y2 <= y1:
            # Якщо bbox вироджений — повернемо нульовий embedding (щоб не падати)
            return np.zeros((128,), dtype=np.float32)

        crop = frame_bgr[y1:y2, x1:x2]
        crops = np.expand_dims(crop, axis=0)  # (1,h,w,3)

        x = preprocess_crops_np(crops, to_rgb=self.cfg.to_rgb)
        emb = self.model.predict(x, verbose=0)[0].astype(np.float32)

        if self.cfg.l2_normalize:
            n = float(np.linalg.norm(emb) + 1e-12)
            emb = emb / n

        return emb
