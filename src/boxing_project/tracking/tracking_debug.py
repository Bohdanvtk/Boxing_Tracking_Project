# src/boxing_project/tracking/tracking_debug.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# -----------------------------
# S U P E R   S I M P L E   L O G
# -----------------------------
@dataclass
class DebugLog:
    """
    Мінімальний логер:
    - може друкувати в консоль (enabled_print=True)
    - може збирати рядки в buffer (для збереження/виводу)
    """
    enabled_print: bool = True
    sink: Callable[[str], None] = print
    buffer: List[str] = field(default_factory=list)

    # будь-які метадані, якщо треба
    meta: Dict[str, Any] = field(default_factory=dict)

    def _emit(self, line: str) -> None:
        self.buffer.append(line)
        if self.enabled_print:
            self.sink(line)

    def section(self, title: str) -> None:
        self._emit("")
        self._emit(title)

    def _print(self, msg: str) -> None:
        self._emit(msg)

    def set_meta(self, cfg: Any, shape: Tuple[int, int]) -> None:
        # не привʼязуємось жорстко до MatchConfig (аби не було імпорт-циклів)
        self.meta = {
            "shape": [int(shape[0]), int(shape[1])],
            "alpha": float(getattr(cfg, "alpha", 0.0)),
            "chi2_gating": float(getattr(cfg, "chi2_gating", 0.0)),
            "large_cost": float(getattr(cfg, "large_cost", 0.0)),
            "min_kp_conf": float(getattr(cfg, "min_kp_conf", 0.0)),
        }


# -----------------------------
# Matcher helpers (API-compatible)
# -----------------------------
def create_matcher_log(cfg: Any, shape: Tuple[int, int], show: bool = True,
                       sink: Optional[Callable[[str], None]] = None) -> DebugLog:
    log = DebugLog(enabled_print=show, sink=sink or print)
    log.set_meta(cfg, shape)
    return log


def make_pair_base(track_index: int, det_index: int) -> Dict[str, Any]:
    return {"track_index": track_index, "det_index": det_index}


def fill_pair_gated_out(*, pair_obj: Dict[str, Any], cfg: Any,
                        d_motion: float = 0.0, d2: Optional[float] = None,
                        cost: Optional[float] = None, **_ignored) -> None:
    """
    Важливо: приймає зайві kwargs (d2, cost, ...) щоб matcher.py не падав.
    """
    large_cost = float(cost if cost is not None else getattr(cfg, "large_cost", 1e6))
    pair_obj["motion"] = {"d2": d2, "allowed": False, "d_motion": float(d_motion)}
    pair_obj["pose"] = {"used_count": 0, "D_pose": 0.0}
    pair_obj["final"] = {
        "alpha": float(getattr(cfg, "alpha", 0.0)),
        "cost": large_cost,
        "components": {"d_motion": float(d_motion), "d_pose": 0.0},
        "reason": "gated_out",
    }


def fill_pair_ok(*, pair_obj: Dict[str, Any], cfg: Any,
                 d_motion: float, d_pose: float, cost: float,
                 pose_dict: Optional[Dict[str, Any]] = None, **_ignored) -> None:
    pair_obj["pose"] = pose_dict or {"used_count": 0, "D_pose": float(d_pose)}
    pair_obj["final"] = {
        "alpha": float(getattr(cfg, "alpha", 0.0)),
        "cost": float(cost),
        "components": {"d_motion": float(d_motion), "d_pose": float(d_pose)},
        "reason": "ok",
    }


def print_gating_result(log: Optional[DebugLog], pair_tag: str, *args, **kwargs) -> None:
    """
    Під matcher.py може прилетіти купа аргументів — ми їх ігноруємо.
    Корисне: d2, chi2, large_cost (якщо є).
    """
    if not (log and log.enabled_print):
        return

    # спробуємо витягнути d2 “розумно”:
    d2 = kwargs.get("d2", None)
    if d2 is None and len(args) >= 1:
        # matcher інколи передає allowed, d2, chi2, large_cost
        # тож d2 може бути другим позиційним
        if len(args) >= 2:
            d2 = args[1]

    chi2 = kwargs.get("chi2_gating", None)
    large_cost = kwargs.get("large_cost", None)

    log.section(f"[pair {pair_tag}] gated out")
    log._print(f"d2={d2}  chi2={chi2}  cost={large_cost}")


def print_pair_result(log: Optional[DebugLog], pair_tag: str, *args, **kwargs) -> None:
    """
    Мінімальний рядок: motion, pose, final_cost.
    Параметри можуть приходити в різних форматах — не ламаємось.
    """
    if not (log and log.enabled_print):
        return

    # типово matcher може викликати: print_pair_result(log, pair_tag, pair_obj, d_motion, d_pose, cost)
    d_motion = None
    d_pose = None
    cost = None

    if len(args) >= 3:
        d_motion = args[-3]
        d_pose = args[-2]
        cost = args[-1]

    log.section(f"[pair {pair_tag}] ok")
    log._print(f"d_motion={d_motion}  d_pose={d_pose}  cost={cost}")


# -----------------------------
# Pose helpers (мінімальні)
# -----------------------------
def set_pose_no_keypoints(pose_dict: Dict[str, Any], log: Optional[DebugLog], pair_tag: str) -> None:
    pose_dict.update({"has_pose": False, "used_count": 0, "D_pose": 0.0})
    if log and log.enabled_print:
        log.section(f"[{pair_tag}] pose")
        log._print("no keypoints -> D_pose=0.0")


def set_pose_no_good_points(pose_dict: Dict[str, Any], log: Optional[DebugLog],
                            pair_tag: str, good_mask: Any) -> None:
    pose_dict.update({"has_pose": True, "used_count": 0, "D_pose": 0.0})
    if log and log.enabled_print:
        log.section(f"[{pair_tag}] pose")
        log._print("no jointly-good keypoints -> D_pose=0.0")


def fill_pose_full_debug(*, pose_dict: Dict[str, Any], log: Optional[DebugLog], pair_tag: str,
                         D_pose: float, used_count: Optional[int] = None, **_ignored) -> None:
    """
    Замість мегатаблиць — тільки коротке резюме.
    """
    if used_count is None:
        # якщо не передали — просто не знаємо
        used_count = int(pose_dict.get("used_count", 0) or 0)

    pose_dict["D_pose"] = float(D_pose)
    pose_dict["used_count"] = int(used_count)

    if log and log.enabled_print:
        log.section(f"[{pair_tag}] pose")
        log._print(f"D_pose={D_pose:.6f} over used_kps={used_count}")


# -----------------------------
# Tracking printing (якщо треба)
# -----------------------------
def print_pre_tracking_results(frame_idx: int) -> None:
    print("\n" + "=" * 80)
    print(f"PRE TRACKING RESULTS: frame={frame_idx}")
    print("=" * 80 + "\n")


def print_tracking_results(log: dict, iteration: int, show_pose_tables: bool = False) -> None:
    """
    Дуже короткий вивід (без полотна таблиць).
    """
    print("\n" + "=" * 80)
    print(f"TRACKING RESULTS: frame={iteration}")
    print("=" * 80)

    active_tracks = log.get("active_tracks", [])
    print(f"active_tracks: {len(active_tracks)}")

    # по 1 рядку на трек
    for t in active_tracks:
        tid = t.get("track_id", "N/A")
        pos = t.get("pos", None)
        print(f"  - Track {tid} pos={pos} logs={len(t.get('match_log', []))}")
