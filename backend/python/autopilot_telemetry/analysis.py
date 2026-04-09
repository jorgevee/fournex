from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any


def summarize_run(events: list[dict[str, Any]]) -> dict[str, Any]:
    per_step = derive_step_metrics(events)
    run_summary = derive_run_summary(events, per_step)
    bottlenecks = classify_bottlenecks(events, per_step, run_summary)
    return {
        "event_count": len(events),
        "step_count": len(per_step),
        "per_step": per_step,
        "run_summary": run_summary,
        "bottlenecks": bottlenecks,
    }


def derive_step_metrics(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: dict[int, dict[str, Any]] = {}

    for event in events:
        step_id = event.get("step_id")
        if step_id is None:
            continue
        step = steps.setdefault(step_id, _empty_step_metrics(step_id))
        _accumulate_step_event(step, event)

    return [steps[step_id] for step_id in sorted(steps)]


def derive_run_summary(events: list[dict[str, Any]], per_step: list[dict[str, Any]]) -> dict[str, Any]:
    gpu_samples = [event for event in events if event.get("event_type") == "gpu_sample"]
    completed_steps = [step for step in per_step if step["status"] == "ok"]
    step_wall_times = [step["step_wall_time_ns"] for step in completed_steps if step["step_wall_time_ns"] > 0]

    avg_gpu_util = _average_numeric_payload(gpu_samples, "utilization_gpu_pct")
    avg_mem_util = _average_numeric_payload(gpu_samples, "utilization_mem_pct")
    peak_memory_ratio = _peak_memory_ratio(gpu_samples)

    total_step_time_ns = sum(step_wall_times)
    throughput_steps_per_sec = 0.0
    if total_step_time_ns > 0:
        throughput_steps_per_sec = len(completed_steps) / (total_step_time_ns / 1_000_000_000)

    return {
        "average_gpu_utilization_pct": avg_gpu_util,
        "average_memory_utilization_pct": avg_mem_util,
        "throughput_steps_per_sec": throughput_steps_per_sec,
        "memory_pressure_peak_ratio": peak_memory_ratio,
        "utilization_instability_pct": _utilization_instability(gpu_samples),
        "step_time_avg_ns": mean(step_wall_times) if step_wall_times else 0,
        "step_time_max_ns": max(step_wall_times) if step_wall_times else 0,
        "shape_volatility_ratio": _shape_volatility_ratio(per_step),
        "profiler_windows_exported": sum(step["profiler_windows_exported"] for step in per_step),
        "dominant_stall_type": _dominant_stall_type(per_step),
    }


def classify_bottlenecks(
    events: list[dict[str, Any]],
    per_step: list[dict[str, Any]],
    run_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    completed_steps = [step for step in per_step if step["status"] == "ok" and step["step_wall_time_ns"] > 0]
    if not completed_steps:
        return []

    classifications: list[dict[str, Any]] = []

    input_ratio = mean(
        _bounded_ratio(step["dataloader_wait_time_ns"], step["step_wall_time_ns"])
        for step in completed_steps
    )
    if input_ratio >= 0.2:
        classifications.append(
            _classification(
                "input_bound",
                input_ratio,
                {
                    "avg_dataloader_fraction": round(input_ratio, 4),
                    "dominant_stall_type": run_summary["dominant_stall_type"],
                },
            )
        )

    copy_ratio = mean(
        _bounded_ratio(step["h2d_copy_time_ns"], step["step_wall_time_ns"])
        for step in completed_steps
    )
    if copy_ratio >= 0.15:
        classifications.append(
            _classification(
                "copy_bound",
                copy_ratio,
                {
                    "avg_h2d_fraction": round(copy_ratio, 4),
                    "steps_with_h2d": sum(1 for step in completed_steps if step["h2d_copy_time_ns"] > 0),
                },
            )
        )

    sync_ratio = mean(
        _bounded_ratio(step["sync_wait_time_ns"], step["step_wall_time_ns"])
        for step in completed_steps
    )
    if sync_ratio >= 0.1:
        classifications.append(
            _classification(
                "sync_bound",
                sync_ratio,
                {
                    "avg_sync_fraction": round(sync_ratio, 4),
                    "steps_with_sync_wait": sum(1 for step in completed_steps if step["sync_wait_time_ns"] > 0),
                },
            )
        )

    if run_summary["average_gpu_utilization_pct"] > 0 and run_summary["average_gpu_utilization_pct"] < 35:
        classifications.append(
            _classification(
                "underutilized_gpu",
                1.0 - (run_summary["average_gpu_utilization_pct"] / 100.0),
                {
                    "average_gpu_utilization_pct": round(run_summary["average_gpu_utilization_pct"], 2),
                    "utilization_instability_pct": round(run_summary["utilization_instability_pct"], 2),
                },
            )
        )

    if run_summary["memory_pressure_peak_ratio"] >= 0.9:
        classifications.append(
            _classification(
                "memory_pressure",
                run_summary["memory_pressure_peak_ratio"],
                {
                    "memory_pressure_peak_ratio": round(run_summary["memory_pressure_peak_ratio"], 4),
                    "average_memory_utilization_pct": round(run_summary["average_memory_utilization_pct"], 2),
                },
            )
        )

    if run_summary["shape_volatility_ratio"] >= 0.3:
        classifications.append(
            _classification(
                "shape_instability",
                run_summary["shape_volatility_ratio"],
                {
                    "shape_volatility_ratio": round(run_summary["shape_volatility_ratio"], 4),
                    "changed_steps": [step["step_id"] for step in per_step if step["shape_changed"]],
                },
            )
        )

    if (
        run_summary["profiler_windows_exported"] > 0
        and run_summary["average_gpu_utilization_pct"] < 50
        and copy_ratio < 0.1
        and input_ratio < 0.1
        and sync_ratio < 0.1
    ):
        classifications.append(
            _classification(
                "launch_bound",
                0.5,
                {
                    "profiler_windows_exported": run_summary["profiler_windows_exported"],
                    "average_gpu_utilization_pct": round(run_summary["average_gpu_utilization_pct"], 2),
                    "note": "Profiler windows were captured but dominant stalls were not input, copy, or sync heavy.",
                },
            )
        )

    classifications.sort(key=lambda item: item["score"], reverse=True)
    return classifications


def _classification(label: str, score: float, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "score": round(score, 4),
        "evidence": evidence,
    }


def _empty_step_metrics(step_id: int) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "status": "unknown",
        "step_kind": None,
        "step_wall_time_ns": 0,
        "dataloader_wait_time_ns": 0,
        "h2d_copy_time_ns": 0,
        "forward_time_ns": 0,
        "backward_time_ns": 0,
        "optimizer_time_ns": 0,
        "sync_wait_time_ns": 0,
        "gpu_active_fraction_proxy": 0.0,
        "shape_signature": None,
        "shape_changed": False,
        "batch_size": 0,
        "sequence_length": None,
        "profiler_windows_exported": 0,
    }


def _accumulate_step_event(step: dict[str, Any], event: dict[str, Any]) -> None:
    event_type = event.get("event_type")
    payload = event.get("payload", {})
    duration_ns = event.get("duration_ns") or 0

    if event_type == "step_start":
        step["step_kind"] = payload.get("step_kind")
        return

    if event_type == "step_end":
        step["status"] = payload.get("status", "unknown")
        step["step_kind"] = payload.get("step_kind", step["step_kind"])
        step["step_wall_time_ns"] = duration_ns
        _update_gpu_active_fraction(step)
        return

    if event_type == "dataloader_span" and payload.get("stage") == "next":
        step["dataloader_wait_time_ns"] += duration_ns
        return

    if event_type == "memcpy_span" and payload.get("copy_kind") == "h2d":
        step["h2d_copy_time_ns"] += duration_ns
        return

    if event_type == "phase_span":
        phase_name = payload.get("phase_name")
        if phase_name == "forward":
            step["forward_time_ns"] = max(step["forward_time_ns"], duration_ns)
        elif phase_name == "backward":
            step["backward_time_ns"] = max(step["backward_time_ns"], duration_ns)
        elif phase_name == "optimizer":
            step["optimizer_time_ns"] = max(step["optimizer_time_ns"], duration_ns)
        return

    if event_type == "sync_wait":
        step["sync_wait_time_ns"] += duration_ns
        return

    if event_type == "shape_snapshot":
        step["batch_size"] = payload.get("batch_size", 0)
        step["sequence_length"] = payload.get("sequence_length")
        step["shape_signature"] = _shape_signature(payload.get("shapes", {}))
        return

    if event_type == "profiler_window" and payload.get("window_state") == "exported":
        step["profiler_windows_exported"] += 1


def _update_gpu_active_fraction(step: dict[str, Any]) -> None:
    step_wall_time_ns = step["step_wall_time_ns"]
    if step_wall_time_ns <= 0:
        step["gpu_active_fraction_proxy"] = 0.0
        return

    active_ns = step["forward_time_ns"] + step["backward_time_ns"] + step["optimizer_time_ns"]
    step["gpu_active_fraction_proxy"] = round(_bounded_ratio(active_ns, step_wall_time_ns), 4)


def _shape_signature(shapes: dict[str, Any]) -> str:
    items = []
    for key in sorted(shapes):
        items.append(f"{key}:{shapes[key]}")
    return "|".join(items)


def _average_numeric_payload(events: list[dict[str, Any]], field: str) -> float:
    values: list[float] = []
    for event in events:
        raw = event.get("payload", {}).get(field)
        number = _to_float(raw)
        if number is not None:
            values.append(number)
    return mean(values) if values else 0.0


def _peak_memory_ratio(gpu_samples: list[dict[str, Any]]) -> float:
    ratios: list[float] = []
    for event in gpu_samples:
        payload = event.get("payload", {})
        used = _to_float(payload.get("memory_used_bytes"))
        total = _to_float(payload.get("memory_total_bytes"))
        if used is not None and total not in (None, 0):
            ratios.append(used / total)
    return max(ratios) if ratios else 0.0


def _utilization_instability(gpu_samples: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for event in gpu_samples:
        number = _to_float(event.get("payload", {}).get("utilization_gpu_pct"))
        if number is not None:
            values.append(number)
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)


def _shape_volatility_ratio(per_step: list[dict[str, Any]]) -> float:
    signatures = [step["shape_signature"] for step in per_step if step["shape_signature"]]
    if len(signatures) < 2:
        for step in per_step:
            step["shape_changed"] = False
        return 0.0

    changes = 0
    previous = signatures[0]
    seen = iter(signatures)
    next(seen)
    for signature in seen:
        changed = signature != previous
        changes += int(changed)
        previous = signature

    previous = None
    for step in per_step:
        signature = step["shape_signature"]
        if not signature:
            step["shape_changed"] = False
            continue
        step["shape_changed"] = previous is not None and signature != previous
        previous = signature

    return changes / max(len(signatures) - 1, 1)


def _dominant_stall_type(per_step: list[dict[str, Any]]) -> str:
    totals = defaultdict(int)
    for step in per_step:
        totals["input_bound"] += step["dataloader_wait_time_ns"]
        totals["copy_bound"] += step["h2d_copy_time_ns"]
        totals["sync_bound"] += step["sync_wait_time_ns"]
        compute_total = step["forward_time_ns"] + step["backward_time_ns"] + step["optimizer_time_ns"]
        totals["compute_bound"] += compute_total

    if not totals:
        return "unknown"

    dominant = max(totals.items(), key=lambda item: item[1])
    return dominant[0] if dominant[1] > 0 else "unknown"


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _bounded_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(float(numerator) / float(denominator), 1.0))
