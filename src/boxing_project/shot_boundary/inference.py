from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .detector import ShotBoundaryDetector, ShotBoundaryConfig


@dataclass(frozen=True)
class ShotBoundaryInferConfig:
    """Configuration for stateful shot-boundary inference."""
    resize_w: int = 160
    resize_h: int = 90
    grid_x: int = 4
    grid_y: int = 4
    ema_alpha: float = 0.2


class ShotBoundaryInferencer:
    """Stateful shot-boundary detector. Lower g means a stronger cut signal."""

    def __init__(self, cfg: ShotBoundaryInferConfig):
        """Create the internal detector and keep its state between frames."""
        sb_cfg = ShotBoundaryConfig(
            resize=(cfg.resize_w, cfg.resize_h),
            grid=(cfg.grid_x, cfg.grid_y),
            ema_alpha=cfg.ema_alpha,
        )
        self.detector = ShotBoundaryDetector(cfg=sb_cfg)

    def update(self, frame_bgr: np.ndarray) -> float:
        """
        Update the detector with a new frame.

        Args:
            frame_bgr: Current OpenCV frame in BGR format, shape (H, W, 3).

        Returns:
            g in [0, 1]. Lower values indicate a stronger cut signal.
        """
        return float(self.detector.update(frame_bgr))
