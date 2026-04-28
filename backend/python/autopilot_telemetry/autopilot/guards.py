from __future__ import annotations

from typing import Any

from .actions import PromotionThresholds, TrialResult


def check_guards(
    result: TrialResult,
    baseline: TrialResult,
    thresholds: PromotionThresholds | None = None,
) -> tuple[bool, list[str]]:
    """
    Returns (passed, failure_reasons).

    Checks the trial result against correctness and promotion guardrails.
    All checks are derived from the trace output — no model internals needed.
    """
    t = thresholds or PromotionThresholds()
    failures: list[str] = []

    if t.require_clean_exit and result.exit_code != 0:
        failures.append(f"non-zero exit code: {result.exit_code}")

    if result.step_count < t.require_sufficient_steps:
        failures.append(
            f"insufficient steps captured: {result.step_count} < {t.require_sufficient_steps}"
        )

    if result.throughput_steps_per_sec <= 0:
        failures.append("zero throughput — workload may not emit step events")

    if result.peak_memory_ratio > t.max_memory_ratio:
        failures.append(
            f"peak memory ratio {result.peak_memory_ratio:.2f} exceeds threshold "
            f"{t.max_memory_ratio:.2f} — possible OOM risk"
        )

    # Step time regression guard: reject if step time got worse by more than threshold.
    if baseline.avg_step_time_ms > 0 and result.avg_step_time_ms > 0:
        regression = (result.avg_step_time_ms - baseline.avg_step_time_ms) / baseline.avg_step_time_ms
        if regression > t.max_step_time_regression:
            failures.append(
                f"step time regressed by {regression:.1%} vs baseline "
                f"({result.avg_step_time_ms:.1f}ms vs {baseline.avg_step_time_ms:.1f}ms)"
            )

    return len(failures) == 0, failures


def extract_metrics_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Pull the metrics we care about from a derived/summary.json payload."""
    scope = summary.get("steady_state") or summary.get("run") or summary
    run_summary = scope.get("run_summary", {})
    step_ns = run_summary.get("step_time_avg_ns", 0) or 0
    return {
        "throughput_steps_per_sec": float(run_summary.get("throughput_steps_per_sec", 0) or 0),
        "avg_gpu_utilization_pct": float(run_summary.get("average_gpu_utilization_pct", 0) or 0),
        "avg_step_time_ms": step_ns / 1_000_000 if step_ns else 0.0,
        "peak_memory_ratio": float(run_summary.get("memory_pressure_peak_ratio", 0) or 0),
        "dominant_stall": str(run_summary.get("dominant_stall_type", "unknown")),
        "step_count": int(scope.get("step_count", 0)),
    }
