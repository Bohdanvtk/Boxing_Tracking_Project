from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .detector import ShotBoundaryDetector, ShotBoundaryConfig


@dataclass(frozen=True)
class ShotBoundaryInferConfig:
    """Конфіг для stateful shot-boundary інференсу (SSIM + EMA)."""
    resize_w: int = 160
    resize_h: int = 90
    grid_x: int = 4
    grid_y: int = 4
    ema_alpha: float = 0.2


class ShotBoundaryInferencer:
    """Stateful shot-boundary: update(frame) -> g (0..1), де g близько 0 означає сильний cut."""

    def __init__(self, cfg: ShotBoundaryInferConfig):
        """Створює внутрішній ShotBoundaryDetector і тримає prev_frame/EMA між кадрами."""
        sb_cfg = ShotBoundaryConfig(
            resize=(cfg.resize_w, cfg.resize_h),
            grid=(cfg.grid_x, cfg.grid_y),
            ema_alpha=cfg.ema_alpha,
        )
        self.detector = ShotBoundaryDetector(cfg=sb_cfg)

    def update(self, frame_bgr: np.ndarray) -> float:
        """
        Оновлює детектор новим кадром.

        Args:
            frame_bgr: поточний кадр (H,W,3) BGR.

        Returns:
            g in [0,1]: довіра до геометрії. Менше g => більше схоже на cut.
        """
        return float(self.detector.update(frame_bgr))
