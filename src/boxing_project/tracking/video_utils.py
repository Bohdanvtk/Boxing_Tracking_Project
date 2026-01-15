from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2


def _resolve(project_root: Path, p: Optional[str | Path]) -> Optional[Path]:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else (project_root / p)


def _list_images(image_dir: Path) -> list[Path]:
    if image_dir is None or not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    images = sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not images:
        raise RuntimeError(f"No images found in directory: {image_dir}")
    return images


def _extract_video_frames(video_path: Path, output_dir: Path) -> list[Path]:
    if video_path is None or not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("frame_*.jpg"):
        old.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: list[Path] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            out_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            frames.append(out_path)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    return frames


def _is_video_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".m4v",
    }


def load_inference_images(data_cfg: dict, project_root: Path) -> list[Path]:
    input_path = _resolve(project_root, data_cfg.get("input_dir"))
    if input_path is None:
        raise KeyError("Missing data.input_dir for inference.")

    if _is_video_path(input_path):
        frames_dir = _resolve(project_root, data_cfg.get("video_frames_dir"))
        if frames_dir is None:
            frames_dir = project_root / ".cache" / "video_frames" / input_path.stem
        return _extract_video_frames(input_path, frames_dir)

    return _list_images(input_path)
