from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort

from .preprocessing import preprocess_crops_np


@dataclass(frozen=True)
class AppearanceEmbedConfig:
    """Конфіг для AppearanceEmbedder (шлях до моделі + препроцес)."""

    model_path: str
    to_rgb: bool = True
    l2_normalize: bool = True


class AppearanceEmbedder:
    """Класовий інференс appearance-embedding: вантажимо модель 1 раз, далі embed() для кожного bbox."""

    def __init__(self, cfg: AppearanceEmbedConfig):
        """Завантажує ONNX appearance encoder з cfg.model_path та тримає його в памʼяті."""
        self.cfg = cfg

        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(Path(cfg.model_path)), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self._embedding_dim = self._infer_embedding_dim_from_meta()
        if self._embedding_dim is None:
            self._embedding_dim = self._infer_embedding_dim_from_dummy()

    def _infer_embedding_dim_from_meta(self) -> int | None:
        shape = self.session.get_outputs()[0].shape
        if not shape or len(shape) < 2:
            return None

        feature_dims = []
        for d in shape[1:]:  # skip batch axis
            if isinstance(d, int) and d > 0:
                feature_dims.append(d)
            else:
                return None

        if not feature_dims:
            return None

        return int(np.prod(feature_dims))

    def _infer_embedding_dim_from_dummy(self) -> int | None:
        try:
            dummy = np.zeros((1, 3, 256, 128), dtype=np.float32)
            out = self.session.run([self.output_name], {self.input_name: dummy})[0]
            return int(np.reshape(out[0], -1).shape[0])
        except Exception:
            return None

    def _zero_embedding(self) -> np.ndarray:
        dim = int(self._embedding_dim or 0)
        return np.zeros((dim,), dtype=np.float32)

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
            return self._zero_embedding()

        crop = frame_bgr[y1:y2, x1:x2]
        crops = np.expand_dims(crop, axis=0)  # (1,h,w,3)

        x = preprocess_crops_np(crops, image_size=(256, 128), to_rgb=self.cfg.to_rgb)
        out = self.session.run([self.output_name], {self.input_name: x})[0]
        emb = np.reshape(out[0], -1).astype(np.float32, copy=False)

        if self._embedding_dim is None:
            self._embedding_dim = int(emb.shape[0])

        if self.cfg.l2_normalize:
            n = float(np.linalg.norm(emb) + 1e-12)
            emb = emb / n

        return emb
