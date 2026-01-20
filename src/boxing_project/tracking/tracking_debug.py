# src/boxing_project/tracking/tracking_debug.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


GENERAL_LOG: list[str] = [] # need to save log in file
FRAME_IDX: int = 1 # inner frame count which indepenable from infer_utils


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

    # optional gating info (if you want)
    allowed: bool = True
    d2: Optional[float] = None


@dataclass
class DebugLog:
    """
    Minimal matrix debugger.

    - create_matrix(n_rows, n_cols): allocates MatrixCell[n_rows][n_cols]
    - log[i, j] returns MatrixCell to fill:
        log[i, j].d_motion = ...
        log[i, j].d_pose = ...
        log[i, j].d_app = ...
        log[i, j].cost = ...
    - show_matrix(): prints ALL (i, j) breakdown + cost matrix table, then resets
    """
    enabled_print: bool = True
    sink: Callable[[str], None] = print

    buffer: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    n_rows: int = 0
    n_cols: int = 0
    matrix: List[List[MatrixCell]] = field(default_factory=list)

    def _emit(self, line: str) -> None:
        self.buffer.append(line)
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

    def show_matrix(self, precision: int = 6) -> None:
        """
        Prints:
          1) Full breakdown for ALL pairs (i, j):
             [i; j]:
               d_motion =
               d_pose =
               d_app =
               cost =
          2) Cost matrix table
        Then resets the matrix.
        """

        global FRAME_IDX

        if not self.enabled_print:
            self.reset_matrix()
            return

        if self.n_rows == 0 or self.n_cols == 0:
            self.section("[debug] empty matrix")
            self.reset_matrix()
            return


        g = None
        if isinstance(self.meta, dict):
            g = self.meta.get("g", None)

        if g is not None:
            self.section(f"[debug] frame coefficient g = {g:.3f}")



        # 1) Breakdown for ALL cells
        self.section("[debug] per-pair breakdown for ALL (i, j)")
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                cell = self.matrix[i][j]
                self.line(f"\n[{i}; {j}]:")
                self.line(f"d_motion = {cell.d_motion:.{precision}f}")
                self.line(f"d_pose   = {cell.d_pose:.{precision}f}")
                self.line(f"d_app    = {cell.d_app:.{precision}f}")
                self.line(f"cost     = {cell.cost:.{precision}f}")

        # 2) Cost matrix table (compact)
        self.section(f"\n[debug] matcher matrix (cost) shape={self.n_rows}x{self.n_cols}")
        header = "      " + " ".join([f"j={j:02d}" for j in range(self.n_cols)])
        self.line(header)

        table_precision = 3
        fmt = f"{{:>{table_precision + 6}.{table_precision}f}}"
        for i in range(self.n_rows):
            row_vals = [fmt.format(float(self.matrix[i][j].cost)) for j in range(self.n_cols)]
            self.line(f"i={i:02d} " + " ".join(row_vals))

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
        tid = t.get("track_id", "N/A")
        pos = t.get("pos", None)
        print(f"  - Track {tid} pos={pos}")