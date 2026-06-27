from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_osnet_transform(height: int = 256, width: int = 128) -> transforms.Compose:
    """
    OSNet/ReID preprocessing: Resize(H,W), ToTensor, and ImageNet normalization.
    The input image is expected to be a RGB PIL image.
    """
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )