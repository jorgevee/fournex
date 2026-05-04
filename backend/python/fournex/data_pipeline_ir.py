from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common_ir import EventRecord


@dataclass(slots=True)
class DataPipelineRecord:
    stage: str
    ts_start_ns: int
    ts_end_ns: int
    batch_size: int | None = None
    num_workers: int | None = None
    prefetch_factor: int | None = None
    pinned_memory: bool | None = None
    bytes_read: int | None = None
    host_id: str | None = None
    process_id: int | None = None
    thread_id: int | None = None
    step_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    correlation_id: str | None = None
    source: str = "python_dataloader"
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataPipelineRecord":
        return cls(
            stage=str(data["stage"]),
            ts_start_ns=int(data["ts_start_ns"]),
            ts_end_ns=int(data["ts_end_ns"]),
            batch_size=_optional_int(data.get("batch_size")),
            num_workers=_optional_int(data.get("num_workers")),
            prefetch_factor=_optional_int(data.get("prefetch_factor")),
            pinned_memory=_optional_bool(data.get("pinned_memory")),
            bytes_read=_optional_int(data.get("bytes_read")),
            host_id=_optional_str(data.get("host_id")),
            process_id=_optional_int(data.get("process_id")),
            thread_id=_optional_int(data.get("thread_id")),
            step_id=_optional_str(data.get("step_id")),
            span_id=_optional_str(data.get("span_id")),
            parent_span_id=_optional_str(data.get("parent_span_id")),
            correlation_id=_optional_str(data.get("correlation_id")),
            source=str(data.get("source", "python_dataloader")),
            attrs=dict(data.get("attrs", {})),
        )


def map_data_pipeline_record_to_ir(
    record: DataPipelineRecord,
    *,
    run_id: str,
    clock_domain: str = "monotonic",
) -> EventRecord:
    duration_ns = max(record.ts_end_ns - record.ts_start_ns, 0)

    return EventRecord(
        event_id=f"data_pipeline_{record.stage}_{record.ts_start_ns}",
        run_id=run_id,
        event_family="data_pipeline",
        event_type=_canonical_event_type(record.stage),
        ts_start_ns=record.ts_start_ns,
        ts_end_ns=record.ts_end_ns,
        duration_ns=duration_ns,
        source=record.source,
        clock_domain=clock_domain,
        host_id=record.host_id,
        process_id=record.process_id,
        thread_id=record.thread_id,
        step_id=record.step_id,
        span_id=record.span_id,
        parent_span_id=record.parent_span_id,
        correlation_id=record.correlation_id,
        attrs={
            "stage": record.stage,
            **({"batch_size": record.batch_size} if record.batch_size is not None else {}),
            **({"num_workers": record.num_workers} if record.num_workers is not None else {}),
            **({"prefetch_factor": record.prefetch_factor} if record.prefetch_factor is not None else {}),
            **({"pinned_memory": record.pinned_memory} if record.pinned_memory is not None else {}),
            **({"bytes_read": record.bytes_read} if record.bytes_read is not None else {}),
            "raw_record": _raw_record_payload(record),
            **record.attrs,
        },
    )


def _canonical_event_type(stage: str) -> str:
    lowered = stage.lower()
    if lowered == "next":
        return "dataloader_wait"
    if lowered == "collate":
        return "collate"
    if lowered == "transfer_ready":
        return "batch_ready"
    if lowered == "decode":
        return "decode"
    if lowered == "transform":
        return "transform"
    if lowered == "dataset_read":
        return "dataset_read"
    return "data_pipeline_event"


def _raw_record_payload(record: DataPipelineRecord) -> dict[str, Any]:
    return {
        "stage": record.stage,
        "ts_start_ns": record.ts_start_ns,
        "ts_end_ns": record.ts_end_ns,
        "batch_size": record.batch_size,
        "num_workers": record.num_workers,
        "prefetch_factor": record.prefetch_factor,
        "pinned_memory": record.pinned_memory,
        "bytes_read": record.bytes_read,
        "host_id": record.host_id,
        "process_id": record.process_id,
        "thread_id": record.thread_id,
        "step_id": record.step_id,
        "span_id": record.span_id,
        "parent_span_id": record.parent_span_id,
        "correlation_id": record.correlation_id,
        "source": record.source,
        "attrs": dict(record.attrs),
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
