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
    # raw absolute values
    d_motion: float = 0.0
    d_pose: float = 0.0
    d_app: float = 0.0

    # absolute raw weighted cost used for Track update decision
    update_cost: float = 0.0

    # row-wise relative penalties
    rel_motion: float = 0.0
    rel_pose: float = 0.0
    rel_app: float = 0.0

    # final row-relative matcher cost
    cost: float = 0.0

    # gating/debug info
    allowed: bool = True
    d2: Optional[float] = None
    motion_ok: bool = True
    pose_ok: bool = True
    app_ok: bool = True


@dataclass
class DebugLog:
    """
    Matrix debugger that can print to console AND/OR collect a log for saving.

    - enabled_print controls ONLY console printing (sink).
    - The log buffer is ALWAYS collected, and is appended to GENERAL_LOG in show_matrix().
    - meta can provide IDs:
        meta["track_ids"] -> list of persistent track_ids aligned with tracks list index i
        meta["det_ids"]   -> list of detection ids aligned with detections list index j
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

    def _get_track_id(self, i: int) -> Any:
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
                    if hasattr(t, "track_id"):
                        return getattr(t, "track_id")
        except Exception:
            pass
        return f"track_index={i}"

    def _get_det_id(self, j: int) -> Any:
        try:
            if isinstance(self.meta, dict):
                det_ids = self.meta.get("det_ids", None)
                if isinstance(det_ids, (list, tuple)) and 0 <= j < len(det_ids):
                    return det_ids[j]

                detections = self.meta.get("detections", None)
                if isinstance(detections, (list, tuple)) and 0 <= j < len(detections):
                    d = detections[j]
                    if isinstance(d, dict):
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

    def _get_det_overlap(self, j: int) -> tuple[float, Any, int]:
        try:
            detections = self.meta.get("detections", None)
            if isinstance(detections, (list, tuple)) and 0 <= j < len(detections):
                det = detections[j]
                meta = getattr(det, "meta", {}) if not isinstance(det, dict) else det.get("meta", {})
                return (
                    float(meta.get("max_overlap_iou", 0.0)),
                    meta.get("max_overlap_det_idx", None),
                    len(meta.get("overlap_relations", []) or []),
                )
        except Exception:
            pass

        return 0.0, None, 0

    def show_matrix(self, precision: int = 6) -> None:
        """
        Output format:
          1) g coefficient + weights
          2) Explanation of indexes
          3) Full breakdown PER TRACK over all detections
          4) Summary: best match per track
          5) Row-relative cost matrix
          6) Absolute update cost matrix
        """
        global FRAME_IDX

        self.buffer.clear()

        if self.n_rows == 0 or self.n_cols == 0:
            self.section("[debug] empty matrix")
            frame_header(FRAME_IDX)
            FRAME_IDX += 1
            GENERAL_LOG.extend(self.buffer)
            self.reset_matrix()
            return

        g = None
        if isinstance(self.meta, dict):
            g = self.meta.get("g", None)

        if g is not None:
            self.section(f"[debug] frame coefficient g = {float(g):.3f}")

        w_motion = self.meta.get("w_motion", None)
        w_pose = self.meta.get("w_pose", None)
        w_app = self.meta.get("w_app", None)
        max_update_cost = self.meta.get("max_update_cost", None)

        if w_motion is not None and w_pose is not None and w_app is not None:
            self.section("[debug] Matching / update cost weights:")
            self.line(f"- w_motion = {float(w_motion):.6f}")
            self.line(f"- w_pose   = {float(w_pose):.6f}")
            self.line(f"- w_app    = {float(w_app):.6f}")
            self.line("")
            self.line("[debug] cost     = row-relative matcher cost")
            self.line("[debug] upd_cost = absolute raw weighted cost")
            self.line(
                "[debug] upd_cost = "
                f"{float(w_motion):.3f}*d_motion + "
                f"{float(w_pose):.3f}*d_pose + "
                f"{float(w_app):.3f}*d_app"
            )

        if max_update_cost is not None:
            self.line(f"[debug] max_update_cost = {float(max_update_cost):.6f}")

        self.section("[info] Meaning of indexes:")
        self.line("- Track#i = i-th element in the current tracks list (0-based list index)")
        self.line("- Det#j   = j-th element in the current detections list (0-based list index)")
        self.line("- track_id = persistent ID stored inside Track")
        self.line("- det_id   = ID stored for Detection (if available; otherwise j)")
        self.line("")
        self.line("[debug] cost is RELATIVE-ONLY after hard gating.")
        self.line("[debug] upd_cost is ABSOLUTE weighted raw cost.")
        self.line("[debug] Raw d_* values are used for gating and debug.")
        self.line("[debug] Lower cost/upd_cost = better. large_cost usually means GATED / INVALID MATCH.")

        self.section("[debug] Detection overlaps:")
        for j in range(self.n_cols):
            max_iou, other_idx, n_rel = self._get_det_overlap(j)
            self.line(
                f"- Det#{j}: max_overlap_iou={max_iou:.3f}, "
                f"max_overlap_det_idx={other_idx}, overlaps={n_rel}"
            )

        self.section("=" * 80)
        self.line("TRACK DETAILS (full breakdown)")
        self.line("=" * 80)

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

                marker = ""
                if j == row_best_j[i]:
                    marker = "   <-- best match for this track (row minimum)"

                reasons = []
                if not bool(cell.allowed):
                    reasons.append("chi2")
                if not bool(cell.motion_ok):
                    reasons.append("motion_thr")
                if not bool(cell.pose_ok):
                    reasons.append("pose_thr")
                if not bool(cell.app_ok):
                    reasons.append("app_thr")

                gated = ""
                if reasons:
                    gated = f"  [GATED: {', '.join(reasons)}]"

                self.line(f"  Det#{j} (det_id={did}):")
                self.line(f"    d_motion = {cell.d_motion:.{precision}f}")
                self.line(f"    d_pose   = {cell.d_pose:.{precision}f}")
                self.line(f"    d_app    = {cell.d_app:.{precision}f}")
                self.line(f"    rel_m    = {cell.rel_motion:.{precision}f}")
                self.line(f"    rel_p    = {cell.rel_pose:.{precision}f}")
                self.line(f"    rel_a    = {cell.rel_app:.{precision}f}")
                if cell.d2 is not None:
                    self.line(f"    d2       = {cell.d2:.{precision}f}")
                self.line(f"    cost     = {cell.cost:.{precision}f}{gated}{marker}")
                self.line(f"    upd_cost = {cell.update_cost:.{precision}f}")

            self.line("")
            self.line("-" * 80)

        self.section("=" * 80)
        self.line("SUMMARY (row-min best match per track)")
        self.line("=" * 80)
        for i in range(self.n_rows):
            tid = self._get_track_id(i)
            bj = row_best_j[i]
            did = self._get_det_id(bj)
            bc = row_best_cost[i]
            self.line(
                f"Track#{i} (track_id={tid}) -> best Det#{bj} "
                f"(det_id={did}) with cost={bc:.{precision}f}"
            )

        self.line("")
        self.line("[info] Note: this is per-track best. Final assignment can differ because one detection")
        self.line("       cannot be matched to many tracks at the same time.")
        self.line("[info] upd_cost is NOT used to choose the match. It is used later to decide Track update.")

        self.section("=" * 80)
        self.line("COST MATRIX (row-relative matcher cost)")
        self.line("=" * 80)

        header = "           " + " ".join([f"Det#{j:02d}" for j in range(self.n_cols)])
        self.line(header)

        table_precision = 3
        fmt = f"{{:>{table_precision + 6}.{table_precision}f}}"
        for i in range(self.n_rows):
            row_vals = [fmt.format(float(self.matrix[i][j].cost)) for j in range(self.n_cols)]
            self.line(f"Track#{i:02d} " + " ".join(row_vals))

        self.section("=" * 80)
        self.line("UPDATE COST MATRIX (absolute raw weighted cost)")
        self.line("=" * 80)

        header = "           " + " ".join([f"Det#{j:02d}" for j in range(self.n_cols)])
        self.line(header)

        table_precision = 3
        fmt = f"{{:>{table_precision + 6}.{table_precision}f}}"
        for i in range(self.n_rows):
            row_vals = [
                fmt.format(float(self.matrix[i][j].update_cost))
                for j in range(self.n_cols)
            ]
            self.line(f"Track#{i:02d} " + " ".join(row_vals))

        birth_debug = self.meta.get("birth_debug", None)
        if isinstance(birth_debug, dict):
            self.section("=" * 80)
            self.line("BIRTH DEBUG")
            self.line("=" * 80)
            for item in birth_debug.get("detections", []):
                det_idx = item.get("det_idx", "N/A")
                self.line(f"Det#{det_idx}:")
                for key in (
                    "action", "reason", "nearest_existing_track_id", "nearest_existing_d_motion",
                    "closeness_status", "pending_id", "required_confirm_hits", "hits",
                    "misses", "age", "birth_score", "d_motion", "pose_status",
                    "pose_penalty", "app_status", "app_penalty", "will_create_new_track",
                ):
                    if key in item:
                        self.line(f"  {key} = {item.get(key)}")
            for conf in birth_debug.get("confirmed", []):
                self.line("")
                self.line("CONFIRMED BIRTH:")
                for key in (
                    "pending_id", "source_det_idx", "reason", "hits", "age",
                    "required_confirm_hits", "will_create_new_track",
                ):
                    if key in conf:
                        self.line(f"  {key} = {conf.get(key)}")

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


def format_track_update_debug_lines(records: List[Dict[str, Any]]) -> List[str]:
    if not records:
        return []
    lines = ["=" * 80, "TRACK UPDATE DEBUG", "=" * 80]
    for rec in records:
        def _cmp(value_key: str, threshold_key: str, flag_key: str) -> str:
            value = float(rec.get(value_key, 0.0))
            threshold = float(rec.get(threshold_key, 0.0))
            op = "<=" if value <= threshold else ">"
            flag = str(bool(rec.get(flag_key, False))).lower()
            return f"{value_key}={value:.6f} {op} {threshold_key}={threshold:.6f} -> {flag_key}={flag}"

        lines.append(f"Track#{rec.get('track_idx')} track_id={rec.get('track_id')} <- Det#{rec.get('det_idx')}:")
        lines.append(f"  {_cmp('d_motion', 'max_update_motion', 'update_motion')}")
        lines.append(f"  {_cmp('d_pose', 'max_update_pose', 'update_pose')}")
        lines.append(f"  {_cmp('d_app', 'max_update_app', 'update_app')}")
        lines.append(f"  row_cost={float(rec.get('row_cost', 0.0)):.6f}")
        lines.append(
            f"  update_cost={float(rec.get('update_cost', 0.0)):.6f} / "
            f"max_update_cost={float(rec.get('max_update_cost', 0.0)):.6f}"
        )
        lines.append(f"  overlap={str(bool(rec.get('track_match_had_overlap', False))).lower()}")
        lines.append(f"  skipped={str(bool(rec.get('track_update_skipped', False))).lower()}")
        lines.append(f"  skip_reason={rec.get('track_update_skip_reason')}")
        lines.append(f"  app_allowed={str(bool(rec.get('track_app_update_allowed', False))).lower()}")
        lines.append(f"  app_block_reason={rec.get('track_app_update_block_reason')}")
    return lines


def format_freeze_debug_lines(records: List[Dict[str, Any]]) -> List[str]:
    if not records:
        return []
    lines = ["=" * 80, "FREEZE DEBUG", "=" * 80]
    for rec in records:
        lines.append(
            f"Track#{rec.get('track_idx')} track_id={rec.get('track_id')}: "
            f"freeze_active={str(bool(rec.get('freeze_active', False))).lower()}, "
            f"frames_left={rec.get('freeze_frames_left')}, "
            f"freeze_sources={rec.get('freeze_sources')}, "
            f"overlap_group_ids={rec.get('overlap_group_ids')}"
        )
    return lines

def append_birth_debug(birth_debug: Dict[str, Any]) -> None:
    if not isinstance(birth_debug, dict):
        return
    GENERAL_LOG.extend(format_birth_debug_lines(birth_debug))


def format_birth_debug_lines(birth_debug: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    if not isinstance(birth_debug, dict):
        return lines

    def _f(value: Any) -> str:
        return "None" if value is None else f"{float(value):.6f}"

    lines.extend(["=" * 80, "BIRTH DEBUG", "=" * 80])
    summary = birth_debug.get("summary", {})
    if isinstance(summary, dict) and summary:
        lines.append(
            "summary: "
            f"incoming_unmatched={summary.get('incoming_unmatched_count', 0)}, "
            f"existing_tracks={summary.get('existing_tracks_count', 0)}, "
            f"pending_before={summary.get('pending_count_before', 0)}, "
            f"confirmed={summary.get('confirmed_count', 0)}, "
            f"pending_after={summary.get('pending_count_after', 0)}"
        )

    for item in birth_debug.get("detections", []):
        det_idx = item.get("det_idx", "N/A")
        center = item.get("det_center")
        lines.append(f"Unmatched Det#{det_idx} center={center}:")
        lines.append(f"  action={item.get('action')}")
        lines.append(f"  reason={item.get('reason')}")
        lines.append("  nearest existing track:")
        lines.append(
            f"    Track#{item.get('nearest_existing_track_idx')} "
            f"track_id={item.get('nearest_existing_track_id')}"
        )
        lines.append(f"    d2={_f(item.get('nearest_existing_d2'))}")
        lines.append(f"    d_motion={_f(item.get('nearest_existing_d_motion'))}")
        lines.append(f"    closeness={item.get('closeness_status')}")
        lines.append(
            f"    thresholds: very_close={_f(item.get('very_close_threshold'))}, "
            f"near={_f(item.get('near_threshold'))}"
        )
        comp = item.get("pending_comparison")
        if isinstance(comp, dict):
            lines.append("  pending comparison:")
            lines.append(f"    pending_id={comp.get('pending_id')}")
            lines.append(f"    d2={_f(comp.get('d2'))}")
            lines.append(f"    d_motion={_f(comp.get('d_motion'))}")
            lines.append(f"    motion_threshold={_f(comp.get('motion_threshold'))}")
            lines.append(f"    motion_passed={str(bool(comp.get('motion_passed', False))).lower()}")
            lines.append(f"    pose_status={comp.get('pose_status')}")
            lines.append(f"    app_status={comp.get('app_status')}")
            if "birth_score" in comp:
                lines.append(f"    birth_score={_f(comp.get('birth_score'))}")
            lines.append(f"    matched_to_pending={str(bool(comp.get('matched_to_pending', False))).lower()}")

    comps = birth_debug.get("pending_comparisons", [])
    if isinstance(comps, list) and comps:
        lines.append("Pending comparisons:")
        for comp in comps:
            lines.append(
                f"  {comp.get('pending_id')} vs Det#{comp.get('det_idx')}: "
                f"d2={_f(comp.get('d2'))}, d_motion={_f(comp.get('d_motion'))}, "
                f"thr={_f(comp.get('motion_threshold'))}, motion_passed={str(bool(comp.get('motion_passed', False))).lower()}, "
                f"pose={comp.get('pose_status')}, app={comp.get('app_status')}, "
                f"score={_f(comp.get('birth_score'))}, matched={str(bool(comp.get('matched_to_pending', False))).lower()}"
            )

    candidates = birth_debug.get("candidates", [])
    if isinstance(candidates, list) and candidates:
        lines.append("Pending candidates:")
        for cand in candidates:
            lines.append(f"  {cand.get('pending_id')}:")
            lines.append(f"    status={cand.get('status')}")
            lines.append(
                f"    hits={cand.get('hits')} misses={cand.get('misses')} age={cand.get('age')} "
                f"required_confirm_hits={cand.get('required_confirm_hits')}"
            )
            lines.append(f"    last_det_idx={cand.get('last_det_idx')}")
            lines.append(f"    last_center={cand.get('last_center')}")

    for conf in birth_debug.get("confirmed", []):
        lines.append("")
        lines.append("CONFIRMED BIRTH:")
        for key in (
            "pending_id", "source_det_idx", "reason", "hits", "age",
            "required_confirm_hits", "will_create_new_track",
        ):
            if key in conf:
                lines.append(f"  {key} = {conf.get(key)}")
    return lines
