"""Entry point: :class:`BoxingResults` loads ``observations.parquet`` once."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import DEFAULT_PARQUET_NAME, REQUIRED_COLUMNS
from .selection import _Query


class BoxingResults(_Query):
    """The whole ``observations.parquet`` loaded once into memory.

    The constructor accepts either the parquet file directly or an output
    directory. For a directory ``output_directory`` it looks for
    ``output_directory/dataset/observations.parquet`` first and
    ``output_directory/observations.parquet`` as a fallback.

    The API is strictly read-only; the parquet file on disk is never written.
    """

    def __init__(self, path: str | Path):
        parquet_path = self._resolve_path(Path(path))
        df = pd.read_parquet(parquet_path)
        self._validate_schema(df, parquet_path)
        super().__init__(df)
        self._path = parquet_path

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        """Accept the parquet file directly or an output/dataset directory."""
        if path.is_file():
            return path
        if path.is_dir():
            candidates = [
                path / "dataset" / DEFAULT_PARQUET_NAME,
                path / DEFAULT_PARQUET_NAME,
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
            raise FileNotFoundError(
                f"Could not find {DEFAULT_PARQUET_NAME} under {path}. "
                f"Looked at: {[str(c) for c in candidates]}"
            )
        raise FileNotFoundError(f"Path does not exist: {path}")

    @staticmethod
    def _validate_schema(df: pd.DataFrame, parquet_path: Path) -> None:
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(
                f"Invalid observations parquet at {parquet_path}. "
                f"Missing required columns: {missing}"
            )

    @property
    def path(self) -> Path:
        return self._path

    @property
    def available_global_ids(self) -> list:
        ids = self._df["global_track_id"].dropna().unique()
        return sorted(int(value) for value in ids)

    @property
    def available_epochs(self) -> list:
        ids = self._df["epoch_id"].dropna().unique()
        return sorted(int(value) for value in ids)

    def __repr__(self) -> str:
        return f"BoxingResults(path={self._path!s}, rows={len(self._df)})"
