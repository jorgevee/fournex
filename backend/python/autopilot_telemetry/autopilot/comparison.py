from __future__ import annotations

import statistics
from typing import Any

from .actions import PromotionThresholds, TrialResult


def aggregate_repeats(
    *,
    config_id: str,
    label: str,
    repeats: list[TrialResult],
    artifacts_path: str,
    env_vars: dict[str, str] | None = None,
) -> TrialResult:
    if not repeats:
        raise ValueError("aggregate_repeats requires at least one repeat")

    if len(repeats) == 1:
        return repeats[0]

    throughput_values = [
        result.throughput_steps_per_sec
        for result in repeats
        if result.throughput_steps_per_sec > 0
    ]
    step_values = [
        result.avg_step_time_ms
        for result in repeats
        if result.avg_step_time_ms > 0
    ]
    gpu_values = [
        result.avg_gpu_utilization_pct
        for result in repeats
        if result.avg_gpu_utilization_pct > 0
    ]
    memory_values = [result.peak_memory_ratio for result in repeats]
    guard_failures = _merged_failures(repeats)
    quality_metrics = _aggregate_quality([result.quality_metrics for result in repeats])

    aggregate = TrialResult(
        config_id=config_id,
        label=label,
        exit_code=0 if all(result.exit_code == 0 for result in repeats) else _first_nonzero_exit(repeats),
        throughput_steps_per_sec=_median(throughput_values),
        avg_gpu_utilization_pct=_median(gpu_values),
        avg_step_time_ms=_median(step_values),
        peak_memory_ratio=max(memory_values) if memory_values else 0.0,
        dominant_stall=_most_common([result.dominant_stall for result in repeats]),
        step_count=min(result.step_count for result in repeats),
        passed_guards=all(result.passed_guards for result in repeats),
        guard_failures=guard_failures,
        env_vars=env_vars or dict(repeats[0].env_vars),
        quality_metrics=quality_metrics,
        repeat_count=len(repeats),
        throughput_values=throughput_values,
        step_time_values_ms=step_values,
        throughput_stddev=_stddev(throughput_values),
        artifacts_path=artifacts_path,
        artifact_paths={
            "metrics": "metrics.json",
            **{
                f"repeat_{index:03d}": f"repeat_{index:03d}/metrics.json"
                for index in range(1, len(repeats) + 1)
            },
        },
    )
    return aggregate


def annotate_noise_comparison(
    baseline: TrialResult,
    trial: TrialResult,
    thresholds: PromotionThresholds | None = None,
) -> None:
    t = thresholds or PromotionThresholds()
    if baseline.throughput_steps_per_sec > 0 and trial.throughput_steps_per_sec > 0:
        trial.throughput_delta = (
            (trial.throughput_steps_per_sec - baseline.throughput_steps_per_sec)
            / baseline.throughput_steps_per_sec
        )

    baseline_noise = _relative_noise(baseline.throughput_values)
    trial_noise = _relative_noise(trial.throughput_values)
    noise_band = max(baseline_noise, trial_noise)
    trial.noise_band = noise_band
    trial.confidence_label = _confidence_label(
        improvement=trial.throughput_delta,
        noise_band=noise_band,
        repeat_count=min(baseline.repeat_count, trial.repeat_count),
        thresholds=t,
    )

    if (
        t.require_above_noise_band
        and trial.repeat_count > 1
        and baseline.repeat_count > 1
        and trial.throughput_delta > 0
        and trial.throughput_delta <= noise_band
    ):
        note = (
            "noise-aware comparison: throughput improvement "
            f"{trial.throughput_delta:.1%} is within noise band {noise_band:.1%}"
        )
        if note not in trial.guard_failures:
            trial.guard_failures.append(note)
        trial.passed_guards = False
        trial.comparison_notes.append(note)


def _relative_noise(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    med = _median(values)
    if med <= 0:
        return 0.0
    return _stddev(values) / med


def _confidence_label(
    *,
    improvement: float,
    noise_band: float,
    repeat_count: int,
    thresholds: PromotionThresholds,
) -> str:
    if repeat_count < 2:
        return "low"
    if improvement <= max(noise_band, thresholds.min_speedup):
        return "inconclusive"
    margin = improvement - noise_band
    if repeat_count >= 3 and margin >= thresholds.min_speedup:
        return "high"
    if margin >= thresholds.min_speedup / 2:
        return "medium"
    return "low"


def _aggregate_quality(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    numeric_by_key: dict[str, list[float]] = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            number = _to_float(value)
            if number is not None:
                numeric_by_key.setdefault(key, []).append(number)
    for key, values in numeric_by_key.items():
        if key in {"nan_count", "inf_count", "loss_count"}:
            result[key] = int(sum(values))
        elif key in {"max_loss", "max_output_abs_diff", "max_output_rel_diff"}:
            result[key] = max(values)
        elif key in {"min_loss"}:
            result[key] = min(values)
        else:
            result[key] = _median(values)
    return result


def _merged_failures(repeats: list[TrialResult]) -> list[str]:
    failures: list[str] = []
    for index, result in enumerate(repeats, start=1):
        for failure in result.guard_failures:
            failures.append(f"repeat {index}: {failure}")
    return failures


def _first_nonzero_exit(repeats: list[TrialResult]) -> int:
    for result in repeats:
        if result.exit_code != 0:
            return result.exit_code
    return 0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _stddev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _most_common(values: list[str]) -> str:
    nonempty = [value for value in values if value]
    if not nonempty:
        return "unknown"
    return max(set(nonempty), key=nonempty.count)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
