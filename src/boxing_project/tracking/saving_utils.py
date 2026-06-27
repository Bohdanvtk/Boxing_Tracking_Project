"""Saving utilities for the staged tracking pipeline.

This module intentionally avoids old per-frame `extra/` outputs and provides
atomic helpers and table writers used by stages.
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically to avoid partial files on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f"{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)




def atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    """Write parquet atomically via temporary file and rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".parquet", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    """Serialize dict as UTF-8 JSON and write atomically."""
    atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))


def save_manifest(stage_dir: Path, manifest: dict[str, Any]) -> None:
    """Persist `manifest.json` for a stage."""
    atomic_write_json(stage_dir / "manifest.json", manifest)


def save_checkpoint(path: Path, state: Any) -> None:
    """Save Python object checkpoint using pickle + atomic write."""
    atomic_write_bytes(path, pickle.dumps(state))


def load_checkpoint(path: Path) -> Any:
    """Load pickled checkpoint object from disk."""
    return pickle.loads(path.read_bytes())


def save_openpose_results_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write preprocessing metadata table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(path, pd.DataFrame(rows))


def save_local_tracks_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write local track assignments table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(path, pd.DataFrame(rows))


def save_tracking_logs(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write tracking logs table when `save_log=True`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(path, pd.DataFrame(rows))


def save_tracks_similarity(*, nodes, sim: np.ndarray, output_path: Path) -> None:
    """Persist global clustering similarity matrix and node index."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), sim=sim.astype(np.float32), nodes=np.asarray(nodes, dtype=np.int32))


def save_global_tracking_debug(*, save_dir: Path, global_debug: dict) -> None:
    """Store optional compact global debug JSON."""
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "global_tracking_debug.json").write_text(
        json.dumps(global_debug or {}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class FragmentExporter:
    """Minimal fragment exporter for segment snapshots used by legacy workflows."""

    def __init__(self, base_dir: Path, min_hits: int):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.min_hits = int(min_hits)

    def save_tracks(self, tracks, frame_idx: int) -> Path:
        """Save a tiny fragment metadata record."""
        output_path = self.base_dir / f"fragment_{frame_idx:06d}.json"
        output_path.write_text(
            json.dumps({"frame_idx": int(frame_idx), "tracks": len(list(tracks or []))}),
            encoding="utf-8",
        )
        return output_path


def save_tracking_outputs(**kwargs):
    """Deprecated legacy API.

    Raises:
        RuntimeError: Always. Use staged saving stages instead.
    """
    raise RuntimeError("save_tracking_outputs is deprecated in staged pipeline. Use LocalDetSavingStage/GlobalSavingStage.")
