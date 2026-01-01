"""
Apperance embedding module.

Goal:
  crop (bbox RGB/BGR image) -> embedding vector (D,)
Training:
  siamese wrapper + contrastive loss on pairs of crops (same/different boxer).
Usage (in inference/tracking):
  e_app = encoder(preprocess(crop))
"""

from .cnn_model import AppearanceCNNConfig, build_appearance_cnn
from .dataset import CropPairs, prepare_contrastive_arrays
from .losses import contrastive_loss
