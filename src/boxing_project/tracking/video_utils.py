from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Optional

import cv2

_FRAME_RE = re.compile(r"frame_(\d+)\.jpg$")


def frames_to_video(frames_dir, out_path, fps, *, codec: str = "mp4v") -> int:
    """Encode sorted ``frame_*.jpg`` from ``frames_dir`` into ``out_path`` (mp4).

    Returns the number of frames written. Streams frame-by-frame (constant
    memory). Frames are ordered by their parsed numeric index; the output size
    is taken from the first frame and mismatched frames are resized. The write
    is atomic (temp file + ``os.replace``).
    """
    frames_dir, out_path = Path(frames_dir), Path(out_path)
    files = sorted(
        frames_dir.glob("frame_*.jpg"),
        key=lambda p: int(m.group(1)) if (m := _FRAME_RE.search(p.name)) else 0,
    )
    if not files:
        return 0

    first = cv2.imread(str(files[0]))
    if first is None:
        raise RuntimeError(f"Unreadable first frame: {files[0]}")
    h, w = first.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".mp4", dir=str(out_path.parent))
    os.close(fd)
    tmp = Path(tmp_name)

    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*codec), float(fps), (w, h))
    if not writer.isOpened():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"cv2.VideoWriter failed (codec={codec}); try another fourcc/container")

    n = 0
    try:
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
            n += 1
    finally:
        writer.release()

    os.replace(tmp, out_path)
    return n


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


def input_fingerprint(data_cfg: dict, project_root: Path) -> dict:
    input_path = _resolve(project_root, data_cfg.get("input_dir"))
    if input_path is None:
        raise KeyError("Missing data.input_dir for inference.")
    input_path = input_path.resolve()

    if not _is_video_path(input_path):
        raise RuntimeError("Restore input fingerprint expects a video file.")

    stat = input_path.stat()
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")
    try:
        return {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


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
