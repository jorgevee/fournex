from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common_ir import AnnotationRecord, MetricRecord


@dataclass(slots=True)
class NvmlSampleRecord:
    timestamp_ns: int
    device_index: int
    utilization_gpu_pct: float | None = None
    utilization_mem_pct: float | None = None
    memory_used_bytes: int | None = None
    memory_total_bytes: int | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    sm_clock_mhz: float | None = None
    mem_clock_mhz: float | None = None
    source: str = "nvml"
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NvmlSampleRecord":
        return cls(
            timestamp_ns=int(data["timestamp_ns"]),
            device_index=int(data["device_index"]),
            utilization_gpu_pct=_optional_float(data.get("utilization_gpu_pct")),
            utilization_mem_pct=_optional_float(data.get("utilization_mem_pct")),
            memory_used_bytes=_optional_int(data.get("memory_used_bytes")),
            memory_total_bytes=_optional_int(data.get("memory_total_bytes")),
            temperature_c=_optional_float(data.get("temperature_c")),
            power_w=_optional_float(data.get("power_w")),
            sm_clock_mhz=_optional_float(data.get("sm_clock_mhz")),
            mem_clock_mhz=_optional_float(data.get("mem_clock_mhz")),
            source=str(data.get("source", "nvml")),
            attrs=dict(data.get("attrs", {})),
        )


def map_nvml_sample_to_ir(
    sample: NvmlSampleRecord,
    *,
    run_id: str,
    host_id: str = "host_0",
    step_id: str | None = None,
    normalizer_source: str | None = None,
) -> tuple[list[MetricRecord], list[AnnotationRecord]]:
    metrics: list[MetricRecord] = []
    annotations: list[AnnotationRecord] = []

    device_id = f"gpu{sample.device_index}"
    source = normalizer_source or sample.source
    raw_payload = _raw_sample_payload(sample)

    def add_metric(name: str, unit: str, value: float | int | None) -> None:
        if value is None:
            return
        metrics.append(
            MetricRecord(
                metric_id=f"{device_id}_{name}_{sample.timestamp_ns}",
                run_id=run_id,
                metric_name=name,
                metric_unit=unit,
                value=float(value),
                ts_ns=sample.timestamp_ns,
                source=source,
                host_id=host_id,
                device_id=device_id,
                step_id=step_id,
                attrs={
                    "raw_sample": raw_payload,
                },
            )
        )

    add_metric("gpu_utilization", "percent", sample.utilization_gpu_pct)
    add_metric("memory_utilization", "percent", sample.utilization_mem_pct)
    add_metric("memory_used_bytes", "bytes", sample.memory_used_bytes)
    add_metric("memory_total_bytes", "bytes", sample.memory_total_bytes)
    add_metric("temperature_c", "celsius", sample.temperature_c)
    add_metric("power_w", "watts", sample.power_w)
    add_metric("sm_clock_mhz", "mhz", sample.sm_clock_mhz)
    add_metric("mem_clock_mhz", "mhz", sample.mem_clock_mhz)

    memory_pressure_ratio = _memory_pressure_ratio(sample)
    if memory_pressure_ratio is not None and memory_pressure_ratio >= 0.9:
        annotations.append(
            AnnotationRecord(
                annotation_id=f"{device_id}_memory_pressure_{sample.timestamp_ns}",
                run_id=run_id,
                annotation_type="device_health",
                target_id=device_id,
                label="memory_pressure",
                score=round(memory_pressure_ratio, 4),
                source=source,
                attrs={
                    "memory_pressure_ratio": round(memory_pressure_ratio, 4),
                    "raw_sample": raw_payload,
                },
            )
        )

    if sample.temperature_c is not None and sample.temperature_c >= 85.0:
        annotations.append(
            AnnotationRecord(
                annotation_id=f"{device_id}_thermal_{sample.timestamp_ns}",
                run_id=run_id,
                annotation_type="device_health",
                target_id=device_id,
                label="thermal_pressure",
                score=round(min(sample.temperature_c / 100.0, 1.0), 4),
                source=source,
                attrs={
                    "temperature_c": sample.temperature_c,
                    "raw_sample": raw_payload,
                },
            )
        )

    return metrics, annotations


def _memory_pressure_ratio(sample: NvmlSampleRecord) -> float | None:
    if sample.memory_used_bytes is None or sample.memory_total_bytes in (None, 0):
        return None
    return float(sample.memory_used_bytes) / float(sample.memory_total_bytes)


def _raw_sample_payload(sample: NvmlSampleRecord) -> dict[str, Any]:
    return {
        "timestamp_ns": sample.timestamp_ns,
        "device_index": sample.device_index,
        "utilization_gpu_pct": sample.utilization_gpu_pct,
        "utilization_mem_pct": sample.utilization_mem_pct,
        "memory_used_bytes": sample.memory_used_bytes,
        "memory_total_bytes": sample.memory_total_bytes,
        "temperature_c": sample.temperature_c,
        "power_w": sample.power_w,
        "sm_clock_mhz": sample.sm_clock_mhz,
        "mem_clock_mhz": sample.mem_clock_mhz,
        "source": sample.source,
        "attrs": dict(sample.attrs),
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
