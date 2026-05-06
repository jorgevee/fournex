from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class BenchmarkWindow:
    warmup_steps: int = 5
    measurement_steps: int = 20
    repeat_count: int = 1
    timeout_s: int = 60

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.measurement_steps <= 0:
            raise ValueError("measurement_steps must be > 0")
        if self.repeat_count <= 0:
            raise ValueError("repeat_count must be > 0")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")

    @property
    def max_steps(self) -> int:
        return self.warmup_steps + self.measurement_steps

    def env_vars(self) -> dict[str, str]:
        return {
            "FRX_TUNE_WARMUP_STEPS": str(self.warmup_steps),
            "FRX_TUNE_MEASURE_STEPS": str(self.measurement_steps),
            "FRX_TUNE_MAX_STEPS": str(self.max_steps),
            "FRX_TUNE_REPEAT_COUNT": str(self.repeat_count),
        }

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def apply_benchmark_window(
    summary: dict[str, Any],
    window: BenchmarkWindow,
) -> dict[str, Any]:
    summary["benchmark_window"] = window.to_dict()
    source_scope = _scope_with_steps(summary)
    if source_scope is None:
        return summary

    selected = _select_measurement_steps(source_scope.get("per_step", []), window)
    source_summary = source_scope.get("run_summary", {})
    summary["measurement_window"] = {
        "scope": {
            "name": "measurement_window",
            "source_scope": source_scope.get("scope", {}).get("name", "run"),
            "step_ids": [step["step_id"] for step in selected],
        },
        "step_count": len(selected),
        "per_step": selected,
        "run_summary": _summarize_steps(selected, source_summary),
    }
    return summary


def _scope_with_steps(summary: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(summary.get("run"), dict) and summary["run"].get("per_step"):
        return summary["run"]
    if summary.get("per_step"):
        return summary
    if isinstance(summary.get("steady_state"), dict) and summary["steady_state"].get("per_step"):
        return summary["steady_state"]
    return None


def _select_measurement_steps(
    per_step: list[dict[str, Any]],
    window: BenchmarkWindow,
) -> list[dict[str, Any]]:
    ordered = sorted(per_step, key=lambda step: int(step.get("step_id", 0)))
    completed = [step for step in ordered if step.get("status", "ok") == "ok"]
    start = window.warmup_steps
    end = start + window.measurement_steps
    return completed[start:end]


def _summarize_steps(
    steps: list[dict[str, Any]],
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    wall_times = [
        int(step.get("step_wall_time_ns", 0) or 0)
        for step in steps
        if int(step.get("step_wall_time_ns", 0) or 0) > 0
    ]
    total_time_ns = sum(wall_times)
    throughput = 0.0
    if total_time_ns > 0:
        throughput = len(wall_times) / (total_time_ns / 1_000_000_000)

    return {
        "average_gpu_utilization_pct": float(source_summary.get("average_gpu_utilization_pct", 0) or 0),
        "average_memory_utilization_pct": float(source_summary.get("average_memory_utilization_pct", 0) or 0),
        "throughput_steps_per_sec": throughput,
        "memory_pressure_peak_ratio": float(source_summary.get("memory_pressure_peak_ratio", 0) or 0),
        "utilization_instability_pct": float(source_summary.get("utilization_instability_pct", 0) or 0),
        "step_time_avg_ns": mean(wall_times) if wall_times else 0,
        "step_time_max_ns": max(wall_times) if wall_times else 0,
        "shape_volatility_ratio": _shape_volatility_ratio(steps),
        "profiler_windows_exported": sum(int(step.get("profiler_windows_exported", 0) or 0) for step in steps),
        "dominant_stall_type": _dominant_stall_type(steps),
        **_quality_summary(steps),
    }


def _shape_volatility_ratio(steps: list[dict[str, Any]]) -> float:
    if not steps:
        return 0.0
    changed = sum(1 for step in steps if bool(step.get("shape_changed")))
    return changed / len(steps)


def _dominant_stall_type(steps: list[dict[str, Any]]) -> str:
    totals = {
        "input_bound": sum(int(step.get("dataloader_wait_time_ns", 0) or 0) for step in steps),
        "copy_bound": sum(int(step.get("h2d_copy_time_ns", 0) or 0) for step in steps),
        "sync_bound": sum(int(step.get("sync_wait_time_ns", 0) or 0) for step in steps),
    }
    if not any(totals.values()):
        return "none"
    return max(totals, key=totals.get)


def _quality_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    losses: list[float] = []
    nan_count = 0
    inf_count = 0
    for step in steps:
        raw = step.get("loss")
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value != value:
            nan_count += 1
        elif value in (float("inf"), float("-inf")):
            inf_count += 1
        else:
            losses.append(value)

    result: dict[str, Any] = {
        "nan_count": nan_count,
        "inf_count": inf_count,
    }
    if losses:
        result.update(
            {
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "loss_slope": losses[-1] - losses[0],
            }
        )
    return result
