from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable

try:
    import psutil
except ImportError:
    psutil = None


RESET = "\033[0m"
BOLD = "\033[1m"

COLORS = {
    "dark": "\033[38;5;240m",
    "muted": "\033[38;5;245m",
    "text": "\033[38;5;231m",
    "info": "\033[38;5;46m",
    "stage": "\033[38;5;214m",
    "bar": "\033[38;5;220m",
    "empty": "\033[38;5;238m",
    "percent": "\033[38;5;228m",
    "time": "\033[38;5;250m",
    "speed": "\033[38;5;215m",
    "warn": "\033[38;5;203m",
    "ram_ok": "\033[38;5;111m",
    "ram_warn": "\033[38;5;220m",
    "ram_bad": "\033[38;5;203m",
    "vram_ok": "\033[38;5;117m",
    "vram_warn": "\033[38;5;220m",
    "vram_bad": "\033[38;5;203m",
    "cpu": "\033[38;5;81m",
    "gpu": "\033[38;5;141m",
    "io": "\033[38;5;178m",
    "mixed": "\033[38;5;215m",
    "op": "\033[38;5;228m",
    "restore": "\033[38;5;141m",
    "clean": "\033[38;5;75m",
}

STAGE_NAMES = {
    "[0/5]": "[0/5] SETUP",
    "[1/5]": "[1/5] PREPROCESS",
    "[2/5]": "[2/5] LOCAL",
    "[3/5]": "[3/5] LOCAL SAVE",
    "[4/5]": "[4/5] GLOBAL CLUST",
    "[5/5]": "[5/5] GLOBAL SAVE",
}

DEVICE_COLORS = {
    "GPU": "gpu",
    "CPU": "cpu",
    "IO": "io",
    "MIXED": "mixed",
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TOKEN_RE = re.compile(r"(?:^|\s|\|){key}\s*=\s*([A-Za-z0-9_/\-]+)", re.IGNORECASE)
BATCH_RE = re.compile(r"batch\s+(\d+)", re.IGNORECASE)


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _gb(value_bytes: int | float) -> str:
    return f"{float(value_bytes) / (1024 ** 3):.1f}G"


def _style_for_pct(pct: float | None, ok: str, warn: str, bad: str) -> str:
    if pct is None:
        return COLORS["muted"] + BOLD
    if pct >= 90:
        return COLORS[bad] + BOLD
    if pct >= 75:
        return COLORS[warn] + BOLD
    return COLORS[ok] + BOLD


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _extract_token(description: str, key: str, default: str = "---") -> str:
    match = re.search(TOKEN_RE.pattern.format(key=re.escape(key)), description, TOKEN_RE.flags)
    return match.group(1).upper() if match else default


def _extract_batch(description: str) -> str | None:
    match = BATCH_RE.search(description)
    return match.group(1) if match else None


def _extract_stage_name(description: str) -> str:
    for token, name in STAGE_NAMES.items():
        if token in description:
            return name
    return description.upper()[:18]


def _smooth_bar(progress: float, width: int, color: Callable[[str, str], str]) -> str:
    progress = max(0.0, min(1.0, float(progress)))
    partials = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]

    exact = progress * width
    full = min(width, int(exact))
    frac = exact - full

    core = "█" * full
    if full < width:
        core += partials[int(frac * 8)]
        core += "─" * max(0, width - len(core))

    colored = "".join(
        color(ch, COLORS["bar"] + BOLD) if ch == "█" or ch in partials[1:] else color(ch, COLORS["empty"])
        for ch in core
    )
    return color("▕", COLORS["dark"]) + colored + color("▏", COLORS["dark"])


class VramReader:
    def __init__(self, refresh_sec: float = 1.25) -> None:
        self.refresh_sec = float(refresh_sec)
        self.text = "VRAM n/a"
        self.pct: float | None = None
        self.last_refresh = 0.0

    def read(self) -> tuple[str, float | None]:
        now = time.time()
        if now - self.last_refresh < self.refresh_sec:
            return self.text, self.pct

        self.last_refresh = now
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=0.35,
            ).strip()
        except (FileNotFoundError, subprocess.SubprocessError, TimeoutError, ValueError):
            self.text, self.pct = "VRAM n/a", None
            return self.text, self.pct

        if not out:
            self.text, self.pct = "VRAM n/a", None
            return self.text, self.pct

        used_mib, total_mib = [float(x.strip()) for x in out.splitlines()[0].split(",")[:2]]
        self.pct = 0.0 if total_mib <= 0 else (used_mib / total_mib) * 100.0
        self.text = f"VRAM {used_mib / 1024.0:.1f}G/{total_mib / 1024.0:.1f}G {self.pct:>3.0f}%"
        return self.text, self.pct


@dataclass
class _TaskState:
    description: str
    total: int
    completed: int = 0
    start_time: float = 0.0
    last_render_time: float = 0.0
    last_completed: int = -1
    closed: bool = False


class RichStageProgress:
    """
    Compact terminal progress bar for the boxing pipeline.

    Public API intentionally stays compatible with the old version:
        add(key, desc, total)
        update(key, completed=..., advance=..., description=..., force=...)
        message(msg)
        warning(msg)
        finish()
    """

    def __init__(
        self,
        enabled: bool = True,
        *,
        restore_mode: bool = False,
        min_interval_sec: float = 0.025,
        min_frame_step: int = 1,
        bar_width: int = 20,
        stage_width: int = 18,
        op_width: int = 12,
        use_color: bool = True,
    ):
        self.enabled = bool(enabled)
        self.restore_mode = bool(restore_mode)
        self.min_interval_sec = float(min_interval_sec)
        self.min_frame_step = int(min_frame_step)
        self.bar_width = int(bar_width)
        self.stage_width = int(stage_width)
        self.op_width = int(op_width)
        self.use_color = bool(use_color)

        self._tasks: dict[str, _TaskState] = {}
        self._active_key: str | None = None
        self._last_line_len = 0
        self._finished = False
        self._frame = 0
        self._vram = VramReader()

        if self.enabled:
            self._write_line(
                self._c("◆ BOXING PIPELINE", COLORS["stage"] + BOLD)
                + self._c("  booting...  ", COLORS["text"])
                + self._mode_text()
            )

    def _c(self, text: str, style: str) -> str:
        return f"{style}{text}{RESET}" if self.use_color else text

    def _mode_text(self) -> str:
        key = "restore" if self.restore_mode else "clean"
        label = "R" if self.restore_mode else "C"
        return self._c(label, COLORS[key] + BOLD)

    def _write_line(self, line: str) -> None:
        if not self.enabled:
            return
        clean = line.replace("\n", " ")
        visible = len(_strip_ansi(clean))
        sys.stdout.write("\r" + clean + (" " * max(0, self._last_line_len - visible)))
        sys.stdout.flush()
        self._last_line_len = visible

    def _clear_line(self) -> None:
        if self.enabled:
            sys.stdout.write("\r" + (" " * max(self._last_line_len, 1)) + "\r")
            sys.stdout.flush()
            self._last_line_len = 0

    def _newline(self) -> None:
        if self.enabled:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._last_line_len = 0

    def _memory_text(self) -> str:
        if psutil is None:
            return self._c("RAM n/a PY n/a", COLORS["muted"] + BOLD)

        mem = psutil.virtual_memory()
        py_used = psutil.Process(os.getpid()).memory_info().rss
        used = mem.total - mem.available
        text = f"RAM {_gb(used)}/{_gb(mem.total)} {mem.percent:>3.0f}% PY {_gb(py_used)}"
        return self._c(text, _style_for_pct(float(mem.percent), "ram_ok", "ram_warn", "ram_bad"))

    def _vram_text(self) -> str:
        text, pct = self._vram.read()
        return self._c(text, _style_for_pct(pct, "vram_ok", "vram_warn", "vram_bad"))

    def _render_task(self, task: _TaskState) -> None:
        self._frame += 1

        elapsed = time.time() - task.start_time
        ratio = task.completed / max(task.total, 1)
        percent = ratio * 100.0
        speed = task.completed / elapsed if task.completed > 0 and elapsed > 0 else 0.0
        remaining = (task.total - task.completed) / max(speed, 1e-9) if speed else 0.0

        desc = task.description
        batch = _extract_batch(desc)

        parts = [
            self._c(["◐", "◓", "◑", "◒"][self._frame % 4], COLORS["stage"] + BOLD),
            self._c(f"{_extract_stage_name(desc):<{self.stage_width}}", COLORS["stage"] + BOLD),
            self._c(f"{_extract_token(desc, 'op')[:self.op_width]:<{self.op_width}}", COLORS["op"] + BOLD),
            self._c(_extract_token(desc, "dev")[:5], COLORS[DEVICE_COLORS.get(_extract_token(desc, "dev"), "muted")] + BOLD),
            self._c(f"F {task.completed:>4}/{task.total:<4}", COLORS["text"] + BOLD),
            self._c(f"B {int(batch):03d}" if batch is not None else "B ---", COLORS["muted"] + BOLD),
            self._c(f"{percent:>5.1f}%", COLORS["percent"] + BOLD),
            _smooth_bar(ratio, self.bar_width, self._c),
            self._c(f"{_duration(elapsed)}<{_duration(remaining)}", COLORS["time"]),
            self._c(f"{speed:>4.1f}fps", COLORS["speed"] + BOLD),
            self._memory_text(),
            self._vram_text(),
            self._mode_text(),
        ]

        self._write_line(("  ").join(parts))

    def _close_task(self, key: str, *, force_complete: bool) -> None:
        task = self._tasks.get(key)
        if task is None or task.closed:
            return

        if force_complete:
            task.completed = task.total
            if key == "setup" or "[0/5]" in task.description:
                task.description = "[0/5] SETUP COMPLETE | op=DONE | dev=CPU"

        self._render_task(task)
        self._newline()
        task.closed = True

    def add(self, key: str, desc: str, total: int) -> None:
        if not self.enabled:
            return

        if self._active_key is not None and self._active_key != key:
            previous = self._tasks.get(self._active_key)
            force = self._active_key == "setup" or bool(previous and previous.completed >= previous.total)
            self._close_task(self._active_key, force_complete=force)

        if key not in self._tasks:
            self._tasks[key] = _TaskState(
                description=str(desc),
                total=max(int(total), 1),
                start_time=time.time(),
            )

        self._active_key = key
        self.update(key, completed=self._tasks[key].completed, description=desc, force=True)

    def update(self, key: str, **kwargs) -> None:
        if not self.enabled:
            return

        task = self._tasks.get(key)
        if task is None or task.closed:
            return

        self._active_key = key
        force = bool(kwargs.pop("force", False))

        description = kwargs.get("description")
        if description is not None:
            task.description = str(description)

        completed = kwargs.get("completed")
        if completed is not None:
            task.completed = int(completed)

        advance = kwargs.get("advance")
        if advance is not None:
            task.completed += int(advance)

        task.completed = max(0, min(task.completed, task.total))

        now = time.time()
        should_render = (
            force
            or task.last_completed < 0
            or task.completed >= task.total
            or now - task.last_render_time >= self.min_interval_sec
            or abs(task.completed - task.last_completed) >= self.min_frame_step
        )
        if not should_render:
            return

        self._render_task(task)
        task.last_render_time = now
        task.last_completed = task.completed

    def _print_event(self, level: str, msg: str, style: str) -> None:
        if not self.enabled:
            return

        self._clear_line()
        print(
            self._c(level, style + BOLD)
            + self._c("  »  ", COLORS["dark"])
            + self._c(str(msg), COLORS["text"] if level == "WARN" else COLORS["info"] + BOLD),
            flush=True,
        )

        if self._active_key is not None:
            self.update(self._active_key, force=True)

    def message(self, msg: str) -> None:
        self._print_event("INFO", msg, COLORS["text"])

    def warning(self, msg: str) -> None:
        self._print_event("WARN", msg, COLORS["warn"])

    def finish(self) -> None:
        if not self.enabled or self._finished:
            return

        self._finished = True
        if self._active_key is not None:
            task = self._tasks.get(self._active_key)
            if task is not None and not task.closed:
                self._render_task(task)
                self._newline()
                task.closed = True

        print(
            self._c("DONE", COLORS["info"] + BOLD)
            + self._c("  »  ", COLORS["dark"])
            + self._c("PIPELINE FINISHED", COLORS["text"] + BOLD),
            flush=True,
        )