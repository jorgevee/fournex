from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common_ir import EventRecord, MetricRecord


@dataclass(slots=True)
class PytorchProfilerTraceEvent:
    name: str
    cat: str
    ph: str
    ts_us: int
    dur_us: int = 0
    pid: int | None = None
    tid: int | None = None
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_trace_event(cls, data: dict[str, Any]) -> "PytorchProfilerTraceEvent":
        return cls(
            name=str(data.get("name", "")),
            cat=str(data.get("cat", "")),
            ph=str(data.get("ph", "X")),
            ts_us=int(data.get("ts", 0)),
            dur_us=int(data.get("dur", 0) or 0),
            pid=_optional_int(data.get("pid")),
            tid=_optional_int(data.get("tid")),
            args=dict(data.get("args", {})),
        )


@dataclass(slots=True)
class PytorchProfilerTrace:
    trace_events: list[PytorchProfilerTraceEvent]
    source_path: str | None = None

    @classmethod
    def from_json_payload(
        cls, payload: dict[str, Any], *, source_path: str | None = None
    ) -> "PytorchProfilerTrace":
        events = [PytorchProfilerTraceEvent.from_trace_event(item) for item in payload.get("traceEvents", [])]
        return cls(trace_events=events, source_path=source_path)


def map_pytorch_profiler_to_ir(
    trace: PytorchProfilerTrace,
    *,
    run_id: str,
    host_id: str = "host_0",
    clock_domain: str = "pytorch_profiler_us",
) -> tuple[list[EventRecord], list[MetricRecord]]:
    events: list[EventRecord] = []
    metrics: list[MetricRecord] = []

    for index, record in enumerate(trace.trace_events):
        if not record.name:
            continue

        if _is_metric_record(record):
            metrics.append(
                MetricRecord(
                    metric_id=f"prof_metric_{index}",
                    run_id=run_id,
                    metric_name=_canonical_metric_name(record),
                    metric_unit=_metric_unit(record),
                    value=float(record.args.get("value", 0.0)),
                    ts_ns=record.ts_us * 1_000,
                    source="pytorch_profiler",
                    clock_domain=clock_domain,
                    host_id=host_id,
                    device_id=_device_id(record),
                    step_id=_step_id(record),
                    attrs={
                        "raw_name": record.name,
                        "raw_category": record.cat,
                        "raw_event": _raw_trace_payload(record),
                    },
                )
            )
            continue

        ts_start_ns = record.ts_us * 1_000
        duration_ns = max(record.dur_us, 0) * 1_000
        ts_end_ns = ts_start_ns + duration_ns

        events.append(
            EventRecord(
                event_id=f"prof_evt_{index}",
                run_id=run_id,
                event_family=_canonical_event_family(record),
                event_type=_canonical_event_type(record),
                ts_start_ns=ts_start_ns,
                ts_end_ns=ts_end_ns,
                duration_ns=duration_ns,
                source="pytorch_profiler",
                clock_domain=clock_domain,
                host_id=host_id,
                process_id=record.pid,
                thread_id=record.tid,
                device_id=_device_id(record),
                step_id=_step_id(record),
                span_id=_optional_str(record.args.get("External id")) or _optional_str(record.args.get("Record function id")),
                parent_span_id=_optional_str(record.args.get("Ev Idx")),
                correlation_id=_optional_str(record.args.get("correlation")) or _optional_str(record.args.get("Correlation ID")),
                attrs={
                    "raw_name": record.name,
                    "raw_category": record.cat,
                    "raw_event": _raw_trace_payload(record),
                    **_canonical_attrs(record),
                },
            )
        )

    return events, metrics


def _canonical_event_family(record: PytorchProfilerTraceEvent) -> str:
    cat = record.cat.lower()
    name = record.name.lower()
    if "kernel" in cat or "cuda_kernel" in cat or "cuda" in cat and "kernel" in name:
        return "kernel"
    if "memcpy" in name or "memory" in cat:
        return "memory"
    if "dataloader" in name or "data" in cat:
        return "data_pipeline"
    if "nccl" in name or "distributed" in cat:
        return "distributed"
    return "cpu"


def _canonical_event_type(record: PytorchProfilerTraceEvent) -> str:
    family = _canonical_event_family(record)
    name = record.name.lower()
    if family == "kernel":
        return "cuda_kernel"
    if family == "memory":
        if "memcpy" in name:
            return "memcpy"
        return "memory_event"
    if family == "data_pipeline":
        if "dataloader" in name:
            return "dataloader_event"
        return "data_pipeline_event"
    if family == "distributed":
        return "collective"
    return "cpu_span"


def _canonical_attrs(record: PytorchProfilerTraceEvent) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if _canonical_event_family(record) == "kernel":
        attrs["kernel_name_raw"] = record.name
        attrs["kernel_class_canonical"] = _kernel_class(record.name)
    if _canonical_event_family(record) == "distributed":
        attrs["collective_op"] = _collective_op(record.name)
    if _canonical_event_family(record) == "memory" and "memcpy" in record.name.lower():
        attrs["memory_op"] = "memcpy"
    return attrs


def _kernel_class(name: str) -> str:
    lowered = name.lower()
    if "gemm" in lowered or "mm" in lowered:
        return "gemm"
    if "attention" in lowered:
        return "attention"
    if "layernorm" in lowered or "layer_norm" in lowered:
        return "layernorm"
    if "reduce" in lowered:
        return "reduction"
    if "copy" in lowered or "memcpy" in lowered:
        return "memcpy"
    return "unknown"


def _collective_op(name: str) -> str:
    lowered = name.lower()
    if "allreduce" in lowered or "all_reduce" in lowered:
        return "all_reduce"
    if "broadcast" in lowered:
        return "broadcast"
    if "reduce_scatter" in lowered:
        return "reduce_scatter"
    if "all_gather" in lowered:
        return "all_gather"
    return "unknown"


def _is_metric_record(record: PytorchProfilerTraceEvent) -> bool:
    if record.ph.upper() == "C":
        return True
    return "value" in record.args and record.dur_us == 0


def _canonical_metric_name(record: PytorchProfilerTraceEvent) -> str:
    lowered = record.name.lower()
    if "util" in lowered:
        return "gpu_utilization"
    if "memory" in lowered:
        return "memory_value"
    return record.name


def _metric_unit(record: PytorchProfilerTraceEvent) -> str:
    unit = record.args.get("unit")
    if unit:
        return str(unit)
    lowered = record.name.lower()
    if "util" in lowered:
        return "percent"
    return "unknown"


def _device_id(record: PytorchProfilerTraceEvent) -> str | None:
    device = record.args.get("device")
    if device is None:
        device = record.args.get("Device Id")
    if device is None:
        return None
    return f"gpu{device}"


def _step_id(record: PytorchProfilerTraceEvent) -> str | None:
    step = record.args.get("step")
    if step is None:
        step = record.args.get("Step")
    if step is None:
        return None
    return f"step_{step}"


def _raw_trace_payload(record: PytorchProfilerTraceEvent) -> dict[str, Any]:
    return {
        "name": record.name,
        "cat": record.cat,
        "ph": record.ph,
        "ts": record.ts_us,
        "dur": record.dur_us,
        "pid": record.pid,
        "tid": record.tid,
        "args": dict(record.args),
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
