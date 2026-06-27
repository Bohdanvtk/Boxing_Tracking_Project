"""
Siamese ReID model wrapper used for OSNet fine-tuning.

The model produces embeddings for two input crops and is trained so that
same-identity boxer pairs are close while different-identity pairs are farther apart.
"""


from __future__ import annotations
import sys
sys.path.append("TransReID")

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchreid


class Identity(nn.Module):
    def forward(self, x):
        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _remove_classifier(model: nn.Module) -> None:
    if hasattr(model, "classifier"):
        try:
            setattr(model, "classifier", Identity())
        except Exception:
            pass
    if hasattr(model, "fc"):
        try:
            setattr(model, "fc", Identity())
        except Exception:
            pass


def _infer_feature_dim(model: nn.Module, device: str, is_transreid: bool = False) -> int:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, 256, 128, device=device)

        if is_transreid:
            cam_label = torch.zeros(x.size(0), dtype=torch.long, device=device)
            out = model(x, cam_label=cam_label)
        else:
            out = model(x)

        if out.ndim != 2:
            out = out.view(out.size(0), -1)
        feat_dim = int(out.size(1))

    if was_training:
        model.train()
    return feat_dim


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def try_unfreeze_last_stage(encoder: nn.Module) -> bool:
    if hasattr(encoder, "conv5"):
        unfreeze_module(getattr(encoder, "conv5"))
        return True

    if hasattr(encoder, "layer4"):
        unfreeze_module(getattr(encoder, "layer4"))
        return True

    children = list(encoder.children())
    if children:
        unfreeze_module(children[-1])
        return True

    return False


@dataclass
class SiameseConfig:
    model_name: str = "osnet_x0_5"
    pretrained: bool = True
    freeze_backbone: bool = False
    allow_unfreeze_last_stage: bool = False


class SiameseOSNet(nn.Module):
    def __init__(self, cfg: SiameseConfig, device: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.is_transreid = (cfg.model_name == "TransReID")

        if self.is_transreid:
            from config import cfg as trans_cfg
            from model import make_model

            trans_cfg.merge_from_file("TransReID/configs/Market/vit_transreid_stride.yml")
            trans_cfg.MODEL.JPM = True

            trans_cfg.TEST.WEIGHT = "TransReID/weights/vit_transreid_market.pth"
            trans_cfg.freeze()

            self.encoder = make_model(trans_cfg, num_class=751, camera_num=6, view_num=5)
            self.encoder.load_param(trans_cfg.TEST.WEIGHT)
            self.encoder.train()
        else:
            self.encoder = torchreid.models.build_model(
                name=cfg.model_name,
                pretrained=cfg.pretrained,
                loss="softmax",
                num_classes=1,
            )
            _remove_classifier(self.encoder)

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(dev)

        feat_dim = _infer_feature_dim(self.encoder, dev, self.is_transreid)

        self.proj = ProjectionHead(feat_dim)
        self.proj.to(dev)

        if cfg.freeze_backbone:
            freeze_all(self.encoder)
            unfreeze_module(self.proj)

            if cfg.allow_unfreeze_last_stage:
                try_unfreeze_last_stage(self.encoder)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_transreid:
            was_training = self.encoder.training
            self.encoder.eval()

            cam_label = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            emb = self.encoder(x, cam_label=cam_label)

            if was_training:
                self.encoder.train()
        else:
            emb = self.encoder(x)

        if emb.ndim != 2:
            emb = emb.view(emb.size(0), -1)

        emb = self.proj(emb)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def forward_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ea = self.forward_once(a)
        eb = self.forward_once(b)
        dist = torch.norm(ea - eb, dim=1, keepdim=True)
        return dist