# src/boxing_project/tracking/tracking_debug.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


GENERAL_LOG: list[str] = []  # saved later into file by inference_utils
FRAME_IDX: int = 1  # inner frame count (independent from infer_utils)


def frame_header(frame_idx: int) -> None:
    GENERAL_LOG.append("\n" + "=" * 80)
    GENERAL_LOG.append(f"FRAME {frame_idx:06d}")
    GENERAL_LOG.append("=" * 80)


@dataclass
class MatrixCell:
    """
    One cell of the cost matrix debug.

    Stores all components you compute for a pair (track_i, det_j).
    """
    d_motion: float = 0.0
    d_pose: float = 0.0
    d_app: float = 0.0
    cost: float = 0.0

    # optional gating info
    allowed: bool = True
    d2: Optional[float] = None


@dataclass
class DebugLog:
    """
    Matrix debugger that can print to console AND/OR collect a log for saving.

    - enabled_print controls ONLY console printing (sink).
    - The log buffer is ALWAYS collected, and is appended to GENERAL_LOG in show_matrix().
    - meta can provide IDs:
        meta["track_ids"] -> list of persistent track_ids aligned with tracks list index i
        meta["det_ids"]   -> list of detection ids aligned with detections list index j
      (fallbacks are supported: meta["tracks"], meta["detections"], etc.)
    """
    enabled_print: bool = True
    sink: Callable[[str], None] = print

    buffer: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    n_rows: int = 0
    n_cols: int = 0
    matrix: List[List[MatrixCell]] = field(default_factory=list)

    def _emit(self, line: str) -> None:
        # Always collect into buffer
        self.buffer.append(line)
        # Print only if enabled_print=True
        if self.enabled_print:
            self.sink(line)

    def section(self, title: str) -> None:
        self._emit("")
        self._emit(str(title))

    def line(self, msg: str) -> None:
        self._emit(str(msg))

    def create_matrix(self, n_rows: int, n_cols: int) -> None:
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.matrix = [[MatrixCell() for _ in range(self.n_cols)] for _ in range(self.n_rows)]

    def reset_matrix(self) -> None:
        self.n_rows = 0
        self.n_cols = 0
        self.matrix = []

    def __getitem__(self, idx: Union[Tuple[int, int], int]) -> Union[MatrixCell, List[MatrixCell]]:
        """
        log[i, j] -> MatrixCell
        log[i]    -> row (list of MatrixCell)
        """
        if isinstance(idx, tuple):
            i, j = idx
            return self.matrix[int(i)][int(j)]
        return self.matrix[int(idx)]

    # ----------------------------
    # Helpers to get ids from meta
    # ----------------------------
    def _get_track_id(self, i: int) -> Any:
        """
        Returns a persistent track_id for row index i, if available.
        Fallback: returns f"track_index={i}" if no info.
        """
        try:
            if isinstance(self.meta, dict):
                track_ids = self.meta.get("track_ids", None)
                if isinstance(track_ids, (list, tuple)) and 0 <= i < len(track_ids):
                    return track_ids[i]

                tracks = self.meta.get("tracks", None)
                if isinstance(tracks, (list, tuple)) and 0 <= i < len(tracks):
                    t = tracks[i]
                    if isinstance(t, dict) and "track_id" in t:
                        return t["track_id"]
                    # allow objects with attribute track_id
                    if hasattr(t, "track_id"):
                        return getattr(t, "track_id")
        except Exception:
            pass
        return f"track_index={i}"

    def _get_det_id(self, j: int) -> Any:
        """
        Returns a detection id for col index j, if available.
        Fallback: returns f"det_index={j}" if no info.
        """
        try:
            if isinstance(self.meta, dict):
                det_ids = self.meta.get("det_ids", None)
                if isinstance(det_ids, (list, tuple)) and 0 <= j < len(det_ids):
                    return det_ids[j]

                detections = self.meta.get("detections", None)
                if isinstance(detections, (list, tuple)) and 0 <= j < len(detections):
                    d = detections[j]
                    if isinstance(d, dict):
                        # try common keys
                        for k in ("det_id", "id", "detection_id", "person_id"):
                            if k in d:
                                return d[k]
                    if hasattr(d, "det_id"):
                        return getattr(d, "det_id")
                    if hasattr(d, "id"):
                        return getattr(d, "id")
        except Exception:
            pass
        return f"det_index={j}"

    def show_matrix(self, precision: int = 6) -> None:
        """
        New output format:
          1) g coefficient + explanation of indices
          2) Full breakdown PER TRACK (track_id) over all detections (det_id)
          3) Summary: best match per track (row-min)
          4) Compact cost matrix

        Collects text into GENERAL_LOG regardless of enabled_print.
        """
        global FRAME_IDX

        # Start a fresh buffer for this call
        self.buffer.clear()

        # If empty matrix: still collect a log entry (useful for saving)
        if self.n_rows == 0 or self.n_cols == 0:
            self.section("[debug] empty matrix")
            frame_header(FRAME_IDX)
            FRAME_IDX += 1
            GENERAL_LOG.extend(self.buffer)
            self.reset_matrix()
            return

        # g coefficient (optional)
        g = None
        if isinstance(self.meta, dict):
            g = self.meta.get("g", None)

        if g is not None:
            self.section(f"[debug] frame coefficient g = {float(g):.3f}")

        # Explain indexes once
        self.section("[info] Meaning of indexes:")
        self.line("- Track#i = i-th element in the current tracks list (0-based list index)")
        self.line("- Det#j   = j-th element in the current detections list (0-based list index)")
        self.line("- track_id = persistent ID stored inside Track (e.g., tracks[i].track_id)")
        self.line("- det_id   = ID stored for Detection (if available; otherwise it can be just j)")
        self.line("")
        self.line("[debug] Pairwise costs (lower = better). 880.0 usually means GATED / INVALID MATCH.")

        # -----------------------------------------
        # 1) Full breakdown grouped by each track row
        # -----------------------------------------
        self.section("=" * 80)
        self.line("TRACK DETAILS (full breakdown)")
        self.line("=" * 80)

        # Precompute row minima for "best match" markers
        row_best_j: List[int] = []
        row_best_cost: List[float] = []
        for i in range(self.n_rows):
            best_j = 0
            best_cost = float(self.matrix[i][0].cost)
            for j in range(1, self.n_cols):
                c = float(self.matrix[i][j].cost)
                if c < best_cost:
                    best_cost = c
                    best_j = j
            row_best_j.append(best_j)
            row_best_cost.append(best_cost)

        for i in range(self.n_rows):
            tid = self._get_track_id(i)
            self.line("")
            self.line(f"Track#{i} (track_id={tid}) vs detections:")

            for j in range(self.n_cols):
                did = self._get_det_id(j)
                cell = self.matrix[i][j]

                # marker: best for the row
                marker = ""
                if j == row_best_j[i]:
                    marker = "   <-- best match for this track (row minimum)"

                # gated marker (heuristic: cost >= 880 or not allowed)
                gated = ""
                if (not bool(cell.allowed)) or float(cell.cost) >= 879.999:
                    gated = "  [GATED / INVALID]"

                self.line(f"  Det#{j} (det_id={did}):")
                self.line(f"    d_motion = {cell.d_motion:.{precision}f}")
                self.line(f"    d_pose   = {cell.d_pose:.{precision}f}")
                self.line(f"    d_app    = {cell.d_app:.{precision}f}")
                self.line(f"    cost     = {cell.cost:.{precision}f}{gated}{marker}")

            self.line("")
            self.line("-" * 80)

        # -----------------------------------------
        # 2) Summary per track: best detection (row min)
        # -----------------------------------------
        self.section("=" * 80)
        self.line("SUMMARY (row-min best match per track)")
        self.line("=" * 80)
        for i in range(self.n_rows):
            tid = self._get_track_id(i)
            bj = row_best_j[i]
            did = self._get_det_id(bj)
            bc = row_best_cost[i]
            self.line(f"Track#{i} (track_id={tid}) -> best Det#{bj} (det_id={did}) with cost={bc:.{precision}f}")

        self.line("")
        self.line("[info] Note: this is per-track best. Final assignment can differ because one detection")
        self.line("       cannot be matched to many tracks at the same time.")

        # -----------------------------------------
        # 3) Compact matrix
        # -----------------------------------------
        self.section("=" * 80)
        self.line("COST MATRIX (tracks x detections)")
        self.line("=" * 80)

        # header with Det#j
        header = "           " + " ".join([f"Det#{j:02d}" for j in range(self.n_cols)])
        self.line(header)

        table_precision = 3
        fmt = f"{{:>{table_precision + 6}.{table_precision}f}}"
        for i in range(self.n_rows):
            row_vals = [fmt.format(float(self.matrix[i][j].cost)) for j in range(self.n_cols)]
            self.line(f"Track#{i:02d} " + " ".join(row_vals))

        # -----------------------------------------
        # Persist into GENERAL_LOG for saving
        # -----------------------------------------
        frame_header(FRAME_IDX)
        FRAME_IDX += 1
        GENERAL_LOG.extend(self.buffer)

        self.reset_matrix()


def print_pre_tracking_results(frame_idx: int) -> None:
    print("\n" + "=" * 80)
    print(f"PRE TRACKING RESULTS: frame={frame_idx}")
    print("=" * 80 + "\n")


def print_tracking_results(log: dict, iteration: int, show_pose_tables: bool = False) -> None:
    print("\n" + "=" * 80)
    print(f"TRACKING RESULTS: frame={iteration}")
    print("=" * 80)
    active_tracks = log.get("active_tracks", [])
    print(f"active_tracks: {len(active_tracks)}")
    for t in active_tracks:
        tid = t.get("track_id", "N/A") if isinstance(t, dict) else getattr(t, "track_id", "N/A")
        pos = t.get("pos", None) if isinstance(t, dict) else getattr(t, "pos", None)
        print(f"  - Track {tid} pos={pos}")
