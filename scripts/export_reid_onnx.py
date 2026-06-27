"""
Export the fine-tuned ReID model to ONNX.

The exported ONNX model can be used by the runtime appearance embedding
inference code during multi-object tracking.
"""



from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import onnx

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from boxing_project.reid_training.siamese_model import SiameseOSNet, SiameseConfig


def parse_args():
    p = argparse.ArgumentParser("Export OSNet embedding model to ONNX")
    p.add_argument("--ckpt_pth", required=True, help="Path to .pth state_dict")
    p.add_argument("--out_onnx", required=True, help="Output .onnx path")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.ckpt_pth)
    out_onnx = Path(args.out_onnx)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    device = "cpu"

    cfg = SiameseConfig(
        model_name="TransReID",
        pretrained=True,  # Initial weights are overwritten by load_state_dict.
        freeze_backbone=False,
        allow_unfreeze_last_stage=False,
    )
    model = SiameseOSNet(cfg, device=device)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    class EmbeddingWrapper(torch.nn.Module):
        def __init__(self, m: SiameseOSNet):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.forward_once(x)

    wrapper = EmbeddingWrapper(model).eval()

    dummy = torch.zeros(1, 3, 256, 128, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_onnx),
        opset_version=18,  # Keep opset high enough for the exported graph.
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        dynamo=False,  # Use the legacy exporter for compatibility.
    )

    onnx_model = onnx.load(str(out_onnx))
    onnx.checker.check_model(onnx_model)
    print(f"Exported and checked ONNX: {out_onnx}")


if __name__ == "__main__":
    main()
