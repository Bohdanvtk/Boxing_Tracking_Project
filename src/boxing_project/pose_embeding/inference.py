from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


try:


    import tensorflow as tf
except Exception:
    tf = None

# IMPORTANT:
# Якщо у тебе інша назва функції нормалізації - зміни імпорт на свою.
from .normalization import normalize_keypoints


@dataclass(frozen=True)
class PoseEmbedConfig:
    """Конфіг для PoseEmbedder (шлях до моделі + дрібні параметри інференсу)."""
    model_path: str
    fill_nan_value: float = 0.0
    l2_normalize: bool = True


class PoseEmbedder:
    """Класовий інференс pose-embedding: модель вантажиться 1 раз, embed() викликається багато разів."""

    def __init__(self, cfg: PoseEmbedConfig):
        """Завантажує pose encoder з cfg.model_path (artifacts) та тримає його в памʼяті."""
        if tf is None:
            raise RuntimeError("TensorFlow не доступний. Перевір встановлення tensorflow у venv.")
        self.cfg = cfg
        self.model = tf.keras.models.load_model(str(Path(cfg.model_path)), compile=False)

    def embed(self, keypoints_xy: np.ndarray, kp_conf: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Рахує pose embedding для однієї людини.

        Args:
            keypoints_xy: np.ndarray форми (K,2) з координатами keypoints.
            kp_conf: optional np.ndarray форми (K,) з confidence. Якщо дано, точки з conf<=0 позначимо NaN.

        Returns:
            np.ndarray форми (D,) — embedding (опційно L2-normalized).
        """
        kps = np.asarray(keypoints_xy, dtype=np.float32)
        if kps.ndim != 2 or kps.shape[1] != 2:
            raise ValueError(f"Очікую keypoints форми (K,2), отримав {kps.shape}")

        if kp_conf is not None:
            conf = np.asarray(kp_conf, dtype=np.float32).reshape(-1)
            if conf.shape[0] != kps.shape[0]:
                raise ValueError(f"kp_conf має мати довжину K={kps.shape[0]}, отримав {conf.shape[0]}")
            bad = conf <= 0.0
            kps = kps.copy()
            kps[bad] = np.nan

        # Нормалізація скелета (центр/масштаб/поворот) — твоя функція
        kps_norm = normalize_keypoints(kps)

        # Замінюємо NaN на fill_nan_value, щоб модель могла порахувати embedding
        kps_norm = np.where(np.isfinite(kps_norm), kps_norm, self.cfg.fill_nan_value).astype(np.float32)

        flat = kps_norm.reshape(1, -1)  # (1, 2K)
        emb = self.model.predict(flat, verbose=0)[0].astype(np.float32)

        if self.cfg.l2_normalize:
            n = float(np.linalg.norm(emb) + 1e-12)
            emb = emb / n

        return emb
