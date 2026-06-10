"""Loss functions used for Siamese ReID fine-tuning."""


from __future__ import annotations

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """
    y = 1 means same identity, so the distance should be small.
    y = 0 means different identity, so the distance should be large.

    Returns one loss value per sample.
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = float(margin)

    def forward(self, dist: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pos = y * (dist ** 2)
        neg = (1.0 - y) * (torch.clamp(self.margin - dist, min=0.0) ** 2)
        return pos + neg