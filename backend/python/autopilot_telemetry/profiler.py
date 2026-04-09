from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sdk import build_runtime_event, emit_event, get_runtime_config

try:
    import torch
except ImportError:
    torch = None


@dataclass(slots=True)
class ProfilerSchedule:
    wait: int = 20
    warmup: int = 2
    record: int = 3
    repeat: int = 0

    @property
    def cycle_length(self) -> int:
        return self.wait + self.warmup + self.record


class SampledProfilerController:
    def __init__(
        self,
        schedule: ProfilerSchedule,
        *,
        output_dir: str | None = None,
        enabled: bool = True,
    ):
        self._schedule = schedule
        self._enabled = enabled
        self._output_dir = output_dir or _default_output_dir()
        self._active_window: dict[str, Any] | None = None
        self._completed_cycles = 0
        self._warning_emitted = False

    def on_step_start(self, step: int) -> None:
        if not self._enabled or self._schedule.cycle_length <= 0:
            return
        if self._schedule.repeat and self._completed_cycles >= self._schedule.repeat:
            return

        offset = step % self._schedule.cycle_length
        cycle_index = step // self._schedule.cycle_length

        if offset == 0:
            self._emit_scheduled_window(step)

        if offset == self._schedule.wait:
            self._start_window(step, cycle_index)

    def on_step_end(self, step: int) -> None:
        if not self._enabled or self._active_window is None:
            return
        if step != self._active_window["end_step"]:
            return

        trace_path, recorded_ops = self._export_summary()
        emit_event(
            build_runtime_event(
                event_type="profiler_window",
                step_id=step,
                payload={
                    "window_state": "completed",
                    "start_step": self._active_window["start_step"],
                    "end_step": self._active_window["end_step"],
                    "recorded_ops": recorded_ops,
                },
            )
        )
        emit_event(
            build_runtime_event(
                event_type="profiler_window",
                step_id=step,
                payload={
                    "window_state": "exported",
                    "start_step": self._active_window["start_step"],
                    "end_step": self._active_window["end_step"],
                    "trace_path": trace_path,
                    "recorded_ops": recorded_ops,
                },
            )
        )
        self._completed_cycles += 1
        self._active_window = None

    def _emit_scheduled_window(self, step: int) -> None:
        start_step = step + self._schedule.wait
        end_step = start_step + self._schedule.warmup + self._schedule.record - 1
        emit_event(
            build_runtime_event(
                event_type="profiler_window",
                step_id=step,
                payload={
                    "window_state": "scheduled",
                    "start_step": start_step,
                    "end_step": end_step,
                },
            )
        )

    def _start_window(self, step: int, cycle_index: int) -> None:
        end_step = step + self._schedule.warmup + self._schedule.record - 1
        self._active_window = {
            "cycle_index": cycle_index,
            "start_step": step,
            "end_step": end_step,
        }
        emit_event(
            build_runtime_event(
                event_type="profiler_window",
                step_id=step,
                payload={
                    "window_state": "started",
                    "start_step": step,
                    "end_step": end_step,
                },
            )
        )
        if torch is None and not self._warning_emitted:
            emit_event(
                build_runtime_event(
                    event_type="warning_annotation",
                    level="warning",
                    payload={
                        "code": "torch_profiler_unavailable",
                        "message": "Sampled profiler is running in metadata-only mode because torch is not installed.",
                    },
                )
            )
            self._warning_emitted = True

    def _export_summary(self) -> tuple[str, int]:
        assert self._active_window is not None

        output_dir = Path(self._output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / f"profiler_window_{self._active_window['cycle_index']:04d}.json"

        summary = {
            "backend": "torch_profiler" if torch is not None else "metadata_only",
            "cycle_index": self._active_window["cycle_index"],
            "start_step": self._active_window["start_step"],
            "end_step": self._active_window["end_step"],
            "recorded_ops": 0,
        }
        trace_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return str(trace_path), 0


_controller: SampledProfilerController | None = None


def configure_sampled_profiler(
    *,
    wait: int = 20,
    warmup: int = 2,
    record: int = 3,
    repeat: int = 0,
    output_dir: str | None = None,
    enabled: bool = True,
) -> SampledProfilerController:
    global _controller
    _controller = SampledProfilerController(
        ProfilerSchedule(wait=wait, warmup=warmup, record=record, repeat=repeat),
        output_dir=output_dir,
        enabled=enabled,
    )
    return _controller


def get_profiler_controller() -> SampledProfilerController | None:
    return _controller


def profiler_step_start(step: int) -> None:
    if _controller is not None:
        _controller.on_step_start(step)


def profiler_step_end(step: int) -> None:
    if _controller is not None:
        _controller.on_step_end(step)


def profiler_window(*args, **kwargs):
    return configure_sampled_profiler(*args, **kwargs)


def _default_output_dir() -> str:
    runtime = get_runtime_config()
    output_path = Path(runtime["output_path"])
    return str(output_path.parent / "profiler")
