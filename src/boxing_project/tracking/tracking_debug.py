from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


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
    Matrix debugger that ONLY collects debug logs in memory.

    Important:
    This class must never print to console.
    Console output is controlled only by pipeline progress UI.
    """
    buffer: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    n_rows: int = 0
    n_cols: int = 0
    matrix: List[List[MatrixCell]] = field(default_factory=list)

    def _emit(self, line: str) -> None:
        # Collect only. Never print.
        self.buffer.append(str(line))


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

    def _get_det_overlap(self, j: int) -> Dict[str, Any]:
        try:
            detections = self.meta.get("detections", None)
            if isinstance(detections, (list, tuple)) and 0 <= j < len(detections):
                det = detections[j]
                meta = getattr(det, "meta", {}) if not isinstance(det, dict) else det.get("meta", {})
                return {
                    "max_iou": float(meta.get("max_overlap_iou", 0.0)),
                    "other_idx": meta.get("max_overlap_det_idx", None),
                    "n_rel": len(meta.get("overlap_relations", []) or []),
                    "min_cdn": meta.get("min_center_dist_norm", None),
                    "cdn_idx": meta.get("center_dist_norm_det_idx", None),
                    "thr": meta.get("active_overlap_threshold", None),
                    "zone": meta.get("adaptive_overlap_zone", None),
                    "enabled": meta.get("adaptive_overlap_enabled", None),
                    "reason": meta.get("adaptive_overlap_reason", None),
                }
        except Exception:
            pass

        return {"max_iou": 0.0, "other_idx": None, "n_rel": 0}

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
            ov = self._get_det_overlap(j)
            min_cdn = ov.get("min_cdn")
            thr = ov.get("thr")
            cdn = "inf" if min_cdn is None or not np.isfinite(float(min_cdn)) else f"{float(min_cdn):.2f}"
            thr_s = "n/a" if thr is None else f"{float(thr):.3f}"
            self.line(
                f"- Det#{j}: max_iou={float(ov.get('max_iou', 0.0)):.3f}, "
                f"other=Det#{ov.get('other_idx')}, rels={ov.get('n_rel', 0)}, "
                f"min_cdn={cdn}->Det#{ov.get('cdn_idx')}, thr={thr_s}, "
                f"zone={ov.get('zone')}, enabled={ov.get('enabled')}, reason={ov.get('reason')}"
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
        lines.append(
            f"  hits: before={rec.get('hits_before_update')} after={rec.get('hits_after_update')}; "
            f"sub_confirmed: before={rec.get('sub_confirmed_before_update')} after={rec.get('sub_confirmed_after_update')}; "
            f"confirmed: before={rec.get('confirmed_before_update')} after={rec.get('confirmed_after_update')}"
        )
        lines.append(f"  {_cmp('d_motion', 'max_update_motion', 'update_motion')}")
        lines.append(f"  {_cmp('d_pose', 'max_update_pose', 'update_pose')}")
        lines.append(f"  {_cmp('d_app', 'max_update_app', 'update_app')}")
        lines.append(f"  row_cost={float(rec.get('row_cost', 0.0)):.6f}")
        lines.append(
            f"  update_cost={float(rec.get('update_cost', 0.0)):.6f} / "
            f"max_update_cost={float(rec.get('max_update_cost', 0.0)):.6f}"
        )
        min_cdn = rec.get("min_center_dist_norm")
        cdn = "inf" if min_cdn is None or not np.isfinite(float(min_cdn)) else f"{float(min_cdn):.2f}"
        lines.append(
            f"  overlap: had={str(bool(rec.get('track_match_had_overlap', False))).lower()}, "
            f"max_iou={float(rec.get('max_overlap_iou', 0.0)):.3f}->Det#{rec.get('max_overlap_det_idx')}, "
            f"min_cdn={cdn}->Det#{rec.get('center_dist_norm_det_idx')}, "
            f"thr={float(rec.get('active_overlap_threshold', 0.0)):.3f}, "
            f"zone={rec.get('adaptive_overlap_zone')}, enabled={rec.get('adaptive_overlap_enabled')}, "
            f"reason={rec.get('adaptive_overlap_reason')}, "
            f"risky={rec.get('track_has_risky_overlap')}, "
            f"risk_count={rec.get('risky_overlap_count')}, "
            f"risk_idxs={rec.get('risky_overlap_det_indices')}, "
            f"max_risk_iou={float(rec.get('max_risky_overlap_iou', 0.0)):.3f}, "
            f"raw_max_iou={float(rec.get('raw_max_overlap_iou', rec.get('max_overlap_iou', 0.0))):.3f}"
        )
        lines.append(f"  skipped={str(bool(rec.get('track_update_skipped', False))).lower()}")
        lines.append(f"  skip_reason={rec.get('track_update_skip_reason')}")
        lines.append(f"  app_allowed={str(bool(rec.get('track_app_update_allowed', False))).lower()}")
        lines.append(f"  app_block_reason={rec.get('track_app_update_block_reason')}")
        # Appearance EMA recovery buffer summary.
        lines.append(
            f"  app_mode={rec.get('app_update_mode')}, "
            f"buffer={rec.get('app_buffer_size_before')}->{rec.get('app_buffer_size_after')}, "
            f"upper_eff={rec.get('app_buffer_upper_eff')}, "
            f"stale={rec.get('app_stale_frames_before')}->{rec.get('app_stale_frames_after')}, "
            f"clear_reason={rec.get('app_buffer_clear_reason')}, "
            f"reject_reason={rec.get('app_buffer_reject_reason')}, "
            f"recovery_applied={rec.get('app_recovery_batch_update_applied')}"
        )
    return lines


def format_used_config_lines(config: Dict[str, Any]) -> List[str]:
    if not isinstance(config, dict) or not config:
        return []
    lines = ["=" * 80, "CONFIG VALUES USED", "=" * 80]
    for key in sorted(config):
        lines.append(f"{key}: {config.get(key)}")
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

def format_removed_tracks_lines(
    pruned: List[Dict[str, Any]],
    dead: List[Dict[str, Any]],
) -> List[str]:
    records = []

    for rec in pruned or []:
        item = dict(rec)
        item["remove_type"] = "pruned"
        records.append(item)

    for rec in dead or []:
        item = dict(rec)
        item["remove_type"] = "dead"
        records.append(item)

    if not records:
        return []

    lines = ["=" * 80, "REMOVED TRACKS", "=" * 80]

    for r in records:
        lines.append(
            f"{r.get('remove_type')} track_id={r.get('track_id')}: "
            f"h={r.get('hits')}, "
            f"a={r.get('age')}, "
            f"tsu={r.get('time_since_update')}, "
            f"sub={str(bool(r.get('sub_confirmed', False))).lower()}, "
            f"conf={str(bool(r.get('confirmed', False))).lower()}, "
            f"reason={r.get('reason')}"
        )

    return lines

def format_birth_debug_lines(birth_debug: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    if not isinstance(birth_debug, dict):
        return lines

    def _f(value: Any) -> str:
        if value is None:
            return "None"
        try:
            return f"{float(value):.6f}"
        except (TypeError, ValueError):
            return str(value)

    def _b(value: Any) -> str:
        return str(bool(value)).lower()

    lines.extend(["=" * 80, "BIRTH DEBUG", "=" * 80])

    summary = birth_debug.get("summary", {})
    if isinstance(summary, dict) and summary:
        lines.append(
            "summary: "
            f"incoming_unmatched={summary.get('incoming_unmatched_count', 0)}, "
            f"existing_tracks={summary.get('existing_tracks_count', 0)}, "
            f"stable_existing_tracks={summary.get('stable_existing_tracks_count', 0)}, "
            f"pending_before={summary.get('pending_count_before', 0)}, "
            f"confirmed={summary.get('confirmed_count', 0)}, "
            f"pending_after={summary.get('pending_count_after', 0)}, "
            f"easy_birth={summary.get('easy_birth_created_count', 0)}/"
            f"{summary.get('easy_birth_track_limit', 0)}"
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
            lines.append(f"    motion_passed={_b(comp.get('motion_passed', False))}")
            lines.append(
                f"    score = d_motion({_f(comp.get('d_motion'))}) "
                f"+ pose_penalty({_f(comp.get('pose_penalty'))}) "
                f"+ app_penalty({_f(comp.get('app_penalty'))}) "
                f"+ near_existing_penalty({_f(comp.get('near_existing_penalty'))})"
            )
            lines.append(f"    pose_status={comp.get('pose_status')}")
            lines.append(f"    pose_penalty={_f(comp.get('pose_penalty'))}")
            lines.append(f"    app_status={comp.get('app_status')}")
            lines.append(f"    app_penalty={_f(comp.get('app_penalty'))}")
            lines.append(f"    near_existing_penalty={_f(comp.get('near_existing_penalty'))}")
            lines.append(f"    birth_score={_f(comp.get('birth_score'))}")
            lines.append(f"    max_birth_score={_f(comp.get('max_birth_score'))}")
            lines.append(f"    score_passed={_b(comp.get('score_passed', False))}")
            lines.append(f"    reject_reason={comp.get('reject_reason')}")
            lines.append(f"    matched_to_pending={_b(comp.get('matched_to_pending', False))}")

    comps = birth_debug.get("pending_comparisons", [])
    if isinstance(comps, list) and comps:
        lines.append("Pending comparisons:")

        for comp in comps:
            hits = comp.get("hits")
            required = comp.get("required_confirm_hits")
            age = comp.get("age")
            max_age = comp.get("max_pending_age")
            misses = comp.get("misses")
            max_misses = comp.get("max_pending_misses")

            motion_cmp = "<=" if bool(comp.get("motion_passed", False)) else ">"
            score = comp.get("birth_score")
            max_score = comp.get("max_birth_score")
            score_gate_enabled = max_score is not None and float(max_score) > 0.0
            score_cmp = "<=" if bool(comp.get("score_passed", False)) else ">"
            score_part = (
                f"score={_f(score)}{score_cmp}{_f(max_score)}"
                if score_gate_enabled
                else f"score={_f(score)}"
            )

            lines.append(
                f"  {comp.get('pending_id')}"
                f"(status={comp.get('status')}, mode={comp.get('birth_mode')}, "
                f"hits={hits}/{required}, misses={misses}/{max_misses}, "
                f"age={age}/{max_age}, last_det={comp.get('last_det_idx')}, "
                f"near_track={comp.get('nearest_existing_track_id')}) "
                f"vs Det#{comp.get('det_idx')}: "
                f"d_motion={_f(comp.get('d_motion'))}{motion_cmp}{_f(comp.get('motion_threshold'))} "
                f"motion={_b(comp.get('motion_passed', False))}, "
                f"pose={comp.get('pose_status')}(+{_f(comp.get('pose_penalty'))}), "
                f"app={comp.get('app_status')}(+{_f(comp.get('app_penalty'))}), "
                f"near_pen={_f(comp.get('near_existing_penalty'))}, "
                f"{score_part}, "
                f"score_passed={_b(comp.get('score_passed', False))}, "
                f"matched={_b(comp.get('matched_to_pending', False))}, "
                f"reject={comp.get('reject_reason')}"
            )

    events = birth_debug.get("candidate_events", [])
    if isinstance(events, list) and events:
        lines.append("Pending events:")

        for event in events:
            ev = event.get("event")
            pid = event.get("pending_id")
            status = event.get("status")
            hits = event.get("hits")
            required = event.get("required_confirm_hits")
            misses = event.get("misses")
            max_misses = event.get("max_pending_misses")
            age = event.get("age")
            max_age = event.get("max_pending_age")

            if ev == "matched_detection":
                lines.append(
                    f"  {pid} event=matched_detection Det#{event.get('matched_det_idx')} "
                    f"status={status} hits={hits}/{required} age={age}/{max_age} "
                    f"score={_f(event.get('matched_birth_score'))} "
                    f"d_motion={_f(event.get('matched_d_motion'))} "
                    f"pose={event.get('matched_pose_status')}(+{_f(event.get('matched_pose_penalty'))}) "
                    f"app={event.get('matched_app_status')}(+{_f(event.get('matched_app_penalty'))}) "
                    f"near_pen={_f(event.get('matched_near_existing_penalty'))} "
                    f"ready={_b(event.get('ready_for_track', False))}"
                )

            elif ev == "missed_this_frame":
                lines.append(
                    f"  {pid} event=missed_this_frame "
                    f"status={status} hits={hits}/{required} "
                    f"misses={misses}/{max_misses} age={age}/{max_age} "
                    f"compared={event.get('compared_detection_count')} "
                    f"motion_passed={event.get('motion_passed_count')} "
                    f"score_passed={event.get('score_passed_count')} "
                    f"best_det=Det#{event.get('best_det_idx')} "
                    f"best_score={_f(event.get('best_birth_score'))} "
                    f"best_reject={event.get('best_reject_reason')}"
                )

            elif ev == "created":
                lines.append(
                    f"  {pid} event=created Det#{event.get('det_idx')} "
                    f"status={status} mode={event.get('birth_mode')} "
                    f"hits={hits}/{required} age={age}/{max_age} "
                    f"near_track={event.get('nearest_existing_track_id')} "
                    f"near_d_motion={_f(event.get('nearest_existing_d_motion'))} "
                    f"very_close_bypass={_b(event.get('very_close_bypass', False))}"
                )

            elif ev == "removed":
                lines.append(
                    f"  {pid} event=removed reason={event.get('reason')} "
                    f"status={status} hits={hits}/{required} "
                    f"misses={misses}/{max_misses} age={age}/{max_age}"
                )

            elif ev == "status_changed":
                lines.append(
                    f"  {pid} event=status_changed "
                    f"{event.get('from_status')} -> {event.get('to_status')} "
                    f"near_track={event.get('nearest_existing_track_id')} "
                    f"near_d_motion={_f(event.get('nearest_existing_d_motion'))}"
                )

            else:
                lines.append(
                    f"  {pid} event={ev} status={status} "
                    f"hits={hits}/{required} misses={misses}/{max_misses} age={age}/{max_age}"
                )

    candidates = birth_debug.get("candidates", [])
    if isinstance(candidates, list) and candidates:
        lines.append("Pending candidates:")

        for cand in candidates:
            lines.append(f"  {cand.get('pending_id')}:")
            lines.append(
                f"    status={cand.get('status')}, "
                f"birth_mode={cand.get('birth_mode')}, "
                f"very_close_bypass={_b(cand.get('very_close_bypass', False))}, "
                f"ready={_b(cand.get('ready_for_track', False))}"
            )
            lines.append(
                f"    hits={cand.get('hits')}/{cand.get('required_confirm_hits')} "
                f"hits_left={cand.get('hits_left_to_track')}, "
                f"misses={cand.get('misses')}/{cand.get('max_pending_misses')}, "
                f"age={cand.get('age')}/{cand.get('max_pending_age')} "
                f"age_left={cand.get('age_left')}"
            )
            lines.append(
                f"    first_seen={cand.get('first_seen_frame')}, "
                f"last_seen={cand.get('last_seen_frame')}, "
                f"last_det_idx={cand.get('last_det_idx')}"
            )
            lines.append(f"    last_center={cand.get('last_center')}")
            lines.append(
                f"    nearest_existing_track_id={cand.get('nearest_existing_track_id')}, "
                f"nearest_existing_d_motion={_f(cand.get('nearest_existing_d_motion'))}"
            )

    for conf in birth_debug.get("confirmed", []):
        lines.append("")
        lines.append("CONFIRMED BIRTH:")

        for key in (
            "pending_id",
            "source_det_idx",
            "reason",
            "status",
            "birth_mode",
            "hits",
            "age",
            "required_confirm_hits",
            "hits_left_to_track",
            "ready_for_track",
            "will_create_new_track",
        ):
            if key in conf:
                lines.append(f"  {key} = {conf.get(key)}")

    return lines

# =========================
# Global tracking debug
# =========================

def _global_node_str(node) -> str:
    """Compact node name for global tracking logs: epoch/local track."""
    return f"e{int(node[0])}:t{int(node[1])}"


def _round4(x: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return round(v, 4)


def build_global_tracking_debug(
    *,
    nodes,
    sim,
    adj,
    labels,
    initial_gids,
    final_mapping,
) -> Dict[str, Any]:
    """
    Build compact global-tracking debug data.

    Intentionally logs only the important global ID decisions:
    - tracks used as graph nodes
    - accepted mutual-kNN edges with cosine similarity
    - final local track -> global id assignment
    """
    nodes = list(nodes or [])
    sim = np.asarray(sim, dtype=np.float32)
    adj = list(adj or [])

    edges: List[Dict[str, Any]] = []
    for i, neigh in enumerate(adj):
        for j, value in neigh.items():
            j = int(j)
            if i >= j:
                continue

            edges.append(
                {
                    "a": _global_node_str(nodes[i]),
                    "b": _global_node_str(nodes[j]),
                    "cosine_sim": _round4(value),
                }
            )

    edges.sort(key=lambda row: float(row.get("cosine_sim", 0.0)), reverse=True)

    assignments: List[Dict[str, Any]] = []
    for i, node in enumerate(nodes):
        node_key = (int(node[0]), int(node[1]))

        spectral_label = None
        if labels is not None and i < len(labels):
            spectral_label = int(labels[i])

        initial_gid = None
        if initial_gids is not None and i < len(initial_gids):
            initial_gid = int(initial_gids[i])

        assignments.append(
            {
                "node": _global_node_str(node_key),
                "epoch_id": int(node_key[0]),
                "local_track_id": int(node_key[1]),
                "spectral_label": spectral_label,
                "initial_global_id": initial_gid,
                "final_global_id": int(final_mapping.get(node_key, -1)),
                "edge_degree": int(len(adj[i])) if i < len(adj) else 0,
            }
        )

    debug: Dict[str, Any] = {
        "summary": {
            "nodes": int(len(nodes)),
            "edges": int(len(edges)),
        },
        "edges": edges,
        "assignments": assignments,
    }

    debug["log_lines"] = format_global_tracking_debug_lines(debug)
    return debug


def format_global_tracking_debug_lines(debug: Dict[str, Any]) -> List[str]:
    """Compact human-readable global tracking log."""
    if not isinstance(debug, dict):
        return []

    summary = debug.get("summary", {})
    lines: List[str] = [
        "=" * 80,
        "GLOBAL TRACKING DEBUG",
        "=" * 80,
        f"[GLOBAL] nodes={summary.get('nodes', 0)} edges={summary.get('edges', 0)}",
    ]

    edges = debug.get("edges", []) or []
    lines.append("")
    if edges:
        lines.append("[EDGES] accepted mutual-kNN edges:")
        for edge in edges:
            lines.append(
                f"- {edge.get('a')} <-> {edge.get('b')} "
                f"cos={float(edge.get('cosine_sim', 0.0)):.4f}"
            )
    else:
        lines.append("[EDGES] no accepted mutual-kNN edges")

    assignments = debug.get("assignments", []) or []
    lines.append("")
    if assignments:
        lines.append("[ASSIGNMENTS] local track -> global id:")
        for row in assignments:
            lines.append(
                f"- {row.get('node')} "
                f"label={row.get('spectral_label')} "
                f"init_gid={row.get('initial_global_id')} "
                f"final_gid={row.get('final_global_id')} "
                f"degree={row.get('edge_degree')}"
            )
    else:
        lines.append("[ASSIGNMENTS] no global assignments")

    return lines