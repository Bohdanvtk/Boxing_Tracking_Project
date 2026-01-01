from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from .ssim_detector import global_ssim
import cv2


@dataclass
class ShotBoundaryConfig:
    resize: Optional[Tuple[int, int]] = (160, 90)  # (width, height) or None
    grid: Tuple[int, int] = (4, 4)                 # (rows, cols)
    ema_alpha: float = 0.2                         # smoothing for g


class ShotBoundaryDetector:
    """
    Returns a single coefficient g in [0,1]:
      g ~ 1 => frames similar (same camera/shot)
      g ~ 0 => frames different (likely cut)
    """

    def __init__(self, cfg: ShotBoundaryConfig):
        self.cfg = cfg
        self.prev_gray: Optional[np.ndarray] = None
        self._g_ema: Optional[float] = None

    def reset(self):
        self.prev_gray = None
        self._g_ema = None

    def update(self, frame: np.ndarray) -> float:
        gray = self._prep_gray(frame)

        # first frame: no comparison possible
        if self.prev_gray is None:
            self.prev_gray = gray
            self._g_ema = 1.0
            return 1.0

        ssim_mean = self._blockwise_ssim_mean(self.prev_gray, gray)  # in ~[0,1]
        g_raw = float(np.clip(ssim_mean, 0.0, 1.0))

        # EMA smoothing on g
        if self._g_ema is None:
            g = g_raw
        else:
            a = float(self.cfg.ema_alpha)
            g = (1.0 - a) * float(self._g_ema) + a * g_raw

        self._g_ema = float(g)
        self.prev_gray = gray
        return float(g)

    # ---------- helpers ----------

    def _prep_gray(self, frame):
        x = frame.astype(np.float32)
        if x.max() > 1.5:
            x /= 255.0

        if x.ndim == 3:
            gray = 0.299 * x[..., 2] + 0.587 * x[..., 1] + 0.114 * x[..., 0]  # BGR → gray
        else:
            gray = x

        if self.cfg.resize is not None:
            w, h = self.cfg.resize
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)

        return np.clip(gray, 0.0, 1.0).astype(np.float32)

    def _blockwise_ssim_mean(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        rows, cols = self.cfg.grid
        H, W = prev_gray.shape

        row_edges = np.linspace(0, H, rows + 1, dtype=int)
        col_edges = np.linspace(0, W, cols + 1, dtype=int)

        s = 0.0
        n = 0

        for r in range(rows):
            rs, re = row_edges[r], row_edges[r + 1]
            for c in range(cols):
                cs, ce = col_edges[c], col_edges[c + 1]

                prev_block = prev_gray[rs:re, cs:ce]
                curr_block = curr_gray[rs:re, cs:ce]

                # avoid tiny blocks instability if tiny blocks they don't count
                if curr_block.size < 16:
                    pass
                else:
                    s += float(global_ssim(curr_block, prev_block))
                n += 1

        return float(s / max(n, 1))
