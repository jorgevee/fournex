from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common_ir import EventRecord


@dataclass(slots=True)
class DistributedCommRecord:
    collective_type: str
    backend: str
    rank: int
    world_size: int
    ts_start_ns: int
    ts_end_ns: int
    tensor_bytes: int | None = None
    communicator_id: str | None = None
    group_name: str | None = None
    stream_id: int | None = None
    overlap_with_compute: float | None = None
    wait_time_ns: int | None = None
    active_time_ns: int | None = None
    host_id: str | None = None
    process_id: int | None = None
    thread_id: int | None = None
    device_id: str | None = None
    step_id: str | None = None
    correlation_id: str | None = None
    source: str = "torch_distributed"
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DistributedCommRecord":
        return cls(
            collective_type=str(data["collective_type"]),
            backend=str(data["backend"]),
            rank=int(data["rank"]),
            world_size=int(data["world_size"]),
            ts_start_ns=int(data["ts_start_ns"]),
            ts_end_ns=int(data["ts_end_ns"]),
            tensor_bytes=_optional_int(data.get("tensor_bytes")),
            communicator_id=_optional_str(data.get("communicator_id")),
            group_name=_optional_str(data.get("group_name")),
            stream_id=_optional_int(data.get("stream_id")),
            overlap_with_compute=_optional_float(data.get("overlap_with_compute")),
            wait_time_ns=_optional_int(data.get("wait_time_ns")),
            active_time_ns=_optional_int(data.get("active_time_ns")),
            host_id=_optional_str(data.get("host_id")),
            process_id=_optional_int(data.get("process_id")),
            thread_id=_optional_int(data.get("thread_id")),
            device_id=_optional_str(data.get("device_id")),
            step_id=_optional_str(data.get("step_id")),
            correlation_id=_optional_str(data.get("correlation_id")),
            source=str(data.get("source", "torch_distributed")),
            attrs=dict(data.get("attrs", {})),
        )


def map_distributed_record_to_ir(
    record: DistributedCommRecord,
    *,
    run_id: str,
    clock_domain: str = "monotonic",
) -> EventRecord:
    normalized_collective = _normalize_collective(record.collective_type)
    duration_ns = max(record.ts_end_ns - record.ts_start_ns, 0)

    return EventRecord(
        event_id=f"dist_{record.rank}_{record.ts_start_ns}_{normalized_collective}",
        run_id=run_id,
        event_family="distributed",
        event_type="collective",
        ts_start_ns=record.ts_start_ns,
        ts_end_ns=record.ts_end_ns,
        duration_ns=duration_ns,
        source=record.source,
        clock_domain=clock_domain,
        host_id=record.host_id,
        process_id=record.process_id,
        thread_id=record.thread_id,
        device_id=record.device_id,
        step_id=record.step_id,
        correlation_id=record.correlation_id,
        attrs={
            "rank": record.rank,
            "world_size": record.world_size,
            "backend": record.backend,
            "collective_op": normalized_collective,
            **({"tensor_bytes": record.tensor_bytes} if record.tensor_bytes is not None else {}),
            **({"communicator_id": record.communicator_id} if record.communicator_id else {}),
            **({"group_name": record.group_name} if record.group_name else {}),
            **({"stream_id": record.stream_id} if record.stream_id is not None else {}),
            **(
                {"overlap_with_compute": record.overlap_with_compute}
                if record.overlap_with_compute is not None
                else {}
            ),
            **({"wait_time_ns": record.wait_time_ns} if record.wait_time_ns is not None else {}),
            **({"active_time_ns": record.active_time_ns} if record.active_time_ns is not None else {}),
            "raw_record": _raw_record_payload(record),
            **record.attrs,
        },
    )


def _normalize_collective(value: str) -> str:
    lowered = value.lower()
    if "allreduce" in lowered or "all_reduce" in lowered:
        return "all_reduce"
    if "broadcast" in lowered:
        return "broadcast"
    if "reduce_scatter" in lowered:
        return "reduce_scatter"
    if "all_gather" in lowered:
        return "all_gather"
    if "gather" in lowered:
        return "gather"
    if "scatter" in lowered:
        return "scatter"
    return "unknown"


def _raw_record_payload(record: DistributedCommRecord) -> dict[str, Any]:
    return {
        "collective_type": record.collective_type,
        "backend": record.backend,
        "rank": record.rank,
        "world_size": record.world_size,
        "ts_start_ns": record.ts_start_ns,
        "ts_end_ns": record.ts_end_ns,
        "tensor_bytes": record.tensor_bytes,
        "communicator_id": record.communicator_id,
        "group_name": record.group_name,
        "stream_id": record.stream_id,
        "overlap_with_compute": record.overlap_with_compute,
        "wait_time_ns": record.wait_time_ns,
        "active_time_ns": record.active_time_ns,
        "host_id": record.host_id,
        "process_id": record.process_id,
        "thread_id": record.thread_id,
        "device_id": record.device_id,
        "step_id": record.step_id,
        "correlation_id": record.correlation_id,
        "source": record.source,
        "attrs": dict(record.attrs),
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
