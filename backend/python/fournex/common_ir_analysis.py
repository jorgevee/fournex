from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from .common_ir import AnnotationRecord, EventRecord, MetricRecord, RunRecord
from .kernel_inspector import summarize_kernel_launches


def summarize_ir_run(run: RunRecord) -> dict[str, Any]:
    run.validate()
    per_step = derive_ir_step_summaries(run.events)
    run_summary = derive_ir_run_summary(run.events, run.metrics, per_step)
    derived_annotations = derive_ir_bottleneck_annotations(run, per_step, run_summary)
    return {
        "run_id": run.run_id,
        "step_count": len(per_step),
        "per_step": per_step,
        "run_summary": run_summary,
        "derived_annotations": [annotation.to_dict() for annotation in derived_annotations],
    }


def derive_ir_step_summaries(events: list[EventRecord]) -> list[dict[str, Any]]:
    steps: dict[str, dict[str, Any]] = {}

    for event in events:
        if not event.step_id:
            continue
        step = steps.setdefault(event.step_id, _empty_step_summary(event.step_id))
        _accumulate_step_event(step, event)

    return [steps[step_id] for step_id in sorted(steps)]


def derive_ir_run_summary(
    events: list[EventRecord],
    metrics: list[MetricRecord],
    per_step: list[dict[str, Any]],
) -> dict[str, Any]:
    gpu_util_values = [metric.value for metric in metrics if metric.metric_name == "gpu_utilization"]
    memory_util_values = [metric.value for metric in metrics if metric.metric_name == "memory_utilization"]
    memory_used_values = [metric.value for metric in metrics if metric.metric_name == "memory_used_bytes"]
    memory_total_values = [metric.value for metric in metrics if metric.metric_name == "memory_total_bytes"]

    step_durations = [step["step_wall_time_ns"] for step in per_step if step["step_wall_time_ns"] > 0]
    total_step_time_ns = sum(step_durations)
    throughput_steps_per_sec = 0.0
    if total_step_time_ns > 0:
        throughput_steps_per_sec = len(step_durations) / (total_step_time_ns / 1_000_000_000)

    memory_pressure_peak_ratio = 0.0
    if memory_used_values and memory_total_values:
        max_total = max(memory_total_values)
        if max_total > 0:
            memory_pressure_peak_ratio = max(memory_used_values) / max_total

    return {
        "average_gpu_utilization_pct": mean(gpu_util_values) if gpu_util_values else 0.0,
        "average_memory_utilization_pct": mean(memory_util_values) if memory_util_values else 0.0,
        "throughput_steps_per_sec": throughput_steps_per_sec,
        "memory_pressure_peak_ratio": memory_pressure_peak_ratio,
        "step_time_avg_ns": mean(step_durations) if step_durations else 0.0,
        "step_time_max_ns": max(step_durations) if step_durations else 0.0,
        "shape_volatility_ratio": _shape_volatility_ratio(per_step),
        "communication_time_ns": sum(step["communication_time_ns"] for step in per_step),
        "data_pipeline_time_ns": sum(step["data_pipeline_time_ns"] for step in per_step),
        "compute_time_ns": sum(step["compute_time_ns"] for step in per_step),
        "kernel_launch_summary": summarize_kernel_launches(events),
    }


def derive_ir_bottleneck_annotations(
    run: RunRecord,
    per_step: list[dict[str, Any]],
    run_summary: dict[str, Any],
) -> list[AnnotationRecord]:
    annotations: list[AnnotationRecord] = []
    step_lookup = {step["step_id"]: step for step in per_step}

    for step in per_step:
        if step["step_wall_time_ns"] <= 0:
            continue
        step_annotations = _classify_step(step, run.run_id)
        annotations.extend(step_annotations)

    if run_summary["memory_pressure_peak_ratio"] >= 0.9:
        annotations.append(
            AnnotationRecord(
                annotation_id=f"{run.run_id}_memory_pressure",
                run_id=run.run_id,
                annotation_type="bottleneck",
                target_id=run.run_id,
                label="memory_pressure",
                score=round(run_summary["memory_pressure_peak_ratio"], 4),
                source="common_ir_rule_engine_v1",
                attrs={
                    "memory_pressure_peak_ratio": round(run_summary["memory_pressure_peak_ratio"], 4),
                },
            )
        )

    if run_summary["shape_volatility_ratio"] >= 0.3:
        annotations.append(
            AnnotationRecord(
                annotation_id=f"{run.run_id}_shape_instability",
                run_id=run.run_id,
                annotation_type="bottleneck",
                target_id=run.run_id,
                label="shape_instability",
                score=round(run_summary["shape_volatility_ratio"], 4),
                source="common_ir_rule_engine_v1",
                attrs={
                    "shape_volatility_ratio": round(run_summary["shape_volatility_ratio"], 4),
                    "changed_steps": [step["step_id"] for step in per_step if step["shape_changed"]],
                },
            )
        )

    return annotations


def _classify_step(step: dict[str, Any], run_id: str) -> list[AnnotationRecord]:
    annotations: list[AnnotationRecord] = []
    step_duration = step["step_wall_time_ns"]
    input_ratio = _bounded_ratio(step["data_pipeline_time_ns"], step_duration)
    communication_ratio = _bounded_ratio(step["communication_time_ns"], step_duration)
    compute_ratio = _bounded_ratio(step["compute_time_ns"], step_duration)

    if input_ratio >= 0.2:
        annotations.append(
            AnnotationRecord(
                annotation_id=f"{run_id}_{step['step_id']}_input_bound",
                run_id=run_id,
                annotation_type="bottleneck",
                target_id=step["step_id"],
                label="input_bound",
                score=round(input_ratio, 4),
                source="common_ir_rule_engine_v1",
                attrs={"input_fraction": round(input_ratio, 4)},
            )
        )

    if communication_ratio >= 0.15:
        annotations.append(
            AnnotationRecord(
                annotation_id=f"{run_id}_{step['step_id']}_communication_bound",
                run_id=run_id,
                annotation_type="bottleneck",
                target_id=step["step_id"],
                label="communication_bound",
                score=round(communication_ratio, 4),
                source="common_ir_rule_engine_v1",
                attrs={"communication_fraction": round(communication_ratio, 4)},
            )
        )

    if compute_ratio >= 0.6 and input_ratio < 0.1 and communication_ratio < 0.1:
        annotations.append(
            AnnotationRecord(
                annotation_id=f"{run_id}_{step['step_id']}_compute_bound",
                run_id=run_id,
                annotation_type="bottleneck",
                target_id=step["step_id"],
                label="compute_bound",
                score=round(compute_ratio, 4),
                source="common_ir_rule_engine_v1",
                attrs={"compute_fraction": round(compute_ratio, 4)},
            )
        )

    return annotations


def _empty_step_summary(step_id: str) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "step_wall_time_ns": 0,
        "compute_time_ns": 0,
        "communication_time_ns": 0,
        "data_pipeline_time_ns": 0,
        "shape_signature": None,
        "shape_changed": False,
    }


def _accumulate_step_event(step: dict[str, Any], event: EventRecord) -> None:
    if event.event_type in {"training_step", "inference_step", "eval_step", "cpu_span"}:
        step["step_wall_time_ns"] = max(step["step_wall_time_ns"], event.duration_ns)

    if event.event_family == "kernel":
        step["compute_time_ns"] += event.duration_ns
    elif event.event_family == "distributed":
        step["communication_time_ns"] += event.duration_ns
    elif event.event_family == "data_pipeline":
        step["data_pipeline_time_ns"] += event.duration_ns

    shape_signature = _shape_signature_from_event(event)
    if shape_signature:
        step["shape_signature"] = shape_signature


def _shape_signature_from_event(event: EventRecord) -> str | None:
    if event.event_family != "cpu":
        return None
    shapes = event.attrs.get("shapes")
    if not isinstance(shapes, dict):
        return None
    items = []
    for key in sorted(shapes):
        items.append(f"{key}:{shapes[key]}")
    return "|".join(items)


def _shape_volatility_ratio(per_step: list[dict[str, Any]]) -> float:
    signatures = [step["shape_signature"] for step in per_step if step["shape_signature"]]
    if len(signatures) < 2:
        return 0.0

    changes = 0
    previous = signatures[0]
    for signature in signatures[1:]:
        if signature != previous:
            changes += 1
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


def _bounded_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(float(numerator) / float(denominator), 1.0))
