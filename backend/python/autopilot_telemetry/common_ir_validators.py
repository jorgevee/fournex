from __future__ import annotations

from typing import Any

from .common_ir import AnnotationRecord, EventRecord, MetricRecord, RunRecord


def validate_event_record(event: EventRecord) -> None:
    event.validate()


def validate_metric_record(metric: MetricRecord) -> None:
    metric.validate()


def validate_annotation_record(annotation: AnnotationRecord) -> None:
    annotation.validate()


def validate_run_record(run: RunRecord) -> None:
    run.validate()


def semantic_warnings_for_run(run: RunRecord) -> list[str]:
    warnings: list[str] = []
    known_step_ids = {event.step_id for event in run.events if event.step_id}

    for metric in run.metrics:
        if metric.step_id and metric.step_id not in known_step_ids:
            warnings.append(f"metric {metric.metric_id} references missing step_id {metric.step_id}")

    for event in run.events:
        if event.event_family == "memory":
            attrs = event.attrs
            if attrs.get("memory_op") == "memcpy":
                if attrs.get("src_device") is None and attrs.get("dst_device") is None:
                    warnings.append(f"memory event {event.event_id} is missing memcpy source/destination spaces")

    return warnings


def validate_run_payload(payload: dict[str, Any]) -> RunRecord:
    run = RunRecord.from_dict(payload)
    return run
