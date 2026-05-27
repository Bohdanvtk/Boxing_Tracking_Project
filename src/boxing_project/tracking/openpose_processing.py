"""OpenPose frame preprocessing helpers shared across pipeline modules."""

from __future__ import annotations

from pathlib import Path

import cv2

op = None


def set_openpose_module(op_module) -> None:
    """Set global pyopenpose module used by preprocess_image."""
    global op
    op = op_module


def preprocess_image(opWrapper, img_path: Path, save_width: int, return_img: bool = False):
    """Read frame, resize to save_width and run OpenPose forward pass."""
    if op is None:
        raise RuntimeError("OpenPose module is not initialized. Call set_openpose_module first.")
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to load image: {img_path}")

    h, w = img.shape[:2]
    if w > save_width:
        scale = save_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    datum = op.Datum()
    datum.cvInputData = img
    datums = op.VectorDatum()
    datums.append(datum)
    opWrapper.emplaceAndPop(datums)

    return (datums[0], img) if return_img else datums[0]
