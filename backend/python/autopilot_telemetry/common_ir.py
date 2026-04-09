from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


WORKLOAD_CLASSES = ("training", "inference", "eval", "preprocessing", "unknown")
MODEL_FAMILIES = ("transformer", "cnn", "recsys", "diffusion", "unknown")
EVENT_FAMILIES = ("kernel", "memory", "cpu", "data_pipeline", "distributed", "metric", "annotation")
MEMORY_OPS = ("alloc", "free", "memcpy", "reserve", "prefetch", "page_migration")
BOTTLENECK_CLASSES = (
    "compute_bound",
    "memory_bound",
    "input_bound",
    "sync_bound",
    "communication_bound",
    "launch_overhead",
    "fragmentation",
    "unknown",
)


@dataclass(slots=True)
class JobInfo:
    job_id: str
    workload_class: str
    status: str
    project_id: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty(self.job_id, "job.job_id")
        _require_enum(self.workload_class, WORKLOAD_CLASSES, "job.workload_class")
        _require_non_empty(self.status, "job.status")


@dataclass(slots=True)
class WorkloadInfo:
    model_family: str = "unknown"
    model_name: str | None = None
    precision_mode: str | None = None
    batch_size: int | None = None
    sequence_length: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_enum(self.model_family, MODEL_FAMILIES, "workload.model_family")
        _require_non_negative_optional_int(self.batch_size, "workload.batch_size")
        _require_non_negative_optional_int(self.sequence_length, "workload.sequence_length")


@dataclass(slots=True)
class EventRecord:
    event_id: str
    run_id: str
    event_family: str
    event_type: str
    ts_start_ns: int
    ts_end_ns: int
    duration_ns: int
    source: str
    attrs: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0.0"
    clock_domain: str = "monotonic"
    host_id: str | None = None
    process_id: int | None = None
    thread_id: int | None = None
    device_id: str | None = None
    step_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    correlation_id: str | None = None

    def validate(self) -> None:
        _require_non_empty(self.schema_version, "event.schema_version")
        _require_non_empty(self.event_id, "event.event_id")
        _require_non_empty(self.run_id, "event.run_id")
        _require_enum(self.event_family, EVENT_FAMILIES, "event.event_family")
        _require_non_empty(self.event_type, "event.event_type")
        _require_non_negative_int(self.ts_start_ns, "event.ts_start_ns")
        _require_non_negative_int(self.ts_end_ns, "event.ts_end_ns")
        _require_non_negative_int(self.duration_ns, "event.duration_ns")
        _require_non_empty(self.source, "event.source")
        if self.ts_end_ns < self.ts_start_ns:
            raise ValueError("event.ts_end_ns must be >= event.ts_start_ns")
        if self.duration_ns != self.ts_end_ns - self.ts_start_ns:
            raise ValueError("event.duration_ns must equal event.ts_end_ns - event.ts_start_ns")
        if self.event_family == "kernel" and not self.device_id:
            raise ValueError("kernel events require event.device_id")
        if self.event_family == "distributed" and self.attrs.get("rank") is None:
            raise ValueError("distributed events require attrs.rank")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventRecord":
        event = cls(**data)
        event.validate()
        return event


@dataclass(slots=True)
class MetricRecord:
    metric_id: str
    run_id: str
    metric_name: str
    metric_unit: str
    value: float
    ts_ns: int
    source: str
    attrs: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0.0"
    clock_domain: str = "monotonic"
    host_id: str | None = None
    device_id: str | None = None
    step_id: str | None = None

    def validate(self) -> None:
        _require_non_empty(self.schema_version, "metric.schema_version")
        _require_non_empty(self.metric_id, "metric.metric_id")
        _require_non_empty(self.run_id, "metric.run_id")
        _require_non_empty(self.metric_name, "metric.metric_name")
        _require_non_empty(self.metric_unit, "metric.metric_unit")
        _require_non_negative_int(self.ts_ns, "metric.ts_ns")
        _require_non_empty(self.source, "metric.source")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricRecord":
        metric = cls(**data)
        metric.validate()
        return metric


@dataclass(slots=True)
class AnnotationRecord:
    annotation_id: str
    run_id: str
    annotation_type: str
    target_id: str
    label: str
    score: float
    source: str
    attrs: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0.0"

    def validate(self) -> None:
        _require_non_empty(self.schema_version, "annotation.schema_version")
        _require_non_empty(self.annotation_id, "annotation.annotation_id")
        _require_non_empty(self.run_id, "annotation.run_id")
        _require_non_empty(self.annotation_type, "annotation.annotation_type")
        _require_non_empty(self.target_id, "annotation.target_id")
        _require_non_empty(self.label, "annotation.label")
        _require_non_empty(self.source, "annotation.source")
        if self.score < 0:
            raise ValueError("annotation.score must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotationRecord":
        annotation = cls(**data)
        annotation.validate()
        return annotation


@dataclass(slots=True)
class RunRecord:
    run_id: str
    job: JobInfo
    workload: WorkloadInfo
    events: list[EventRecord] = field(default_factory=list)
    metrics: list[MetricRecord] = field(default_factory=list)
    annotations: list[AnnotationRecord] = field(default_factory=list)
    schema_version: str = "1.0.0"
    collector_version: str | None = None
    normalizer_version: str | None = None
    source_coverage: dict[str, bool] = field(default_factory=dict)
    collection_errors: list[str] = field(default_factory=list)
    sampling_rate: dict[str, float] = field(default_factory=dict)
    dropped_event_counts: dict[str, int] = field(default_factory=dict)

    def validate(self) -> None:
        _require_non_empty(self.schema_version, "run.schema_version")
        _require_non_empty(self.run_id, "run.run_id")
        self.job.validate()
        self.workload.validate()
        for event in self.events:
            event.validate()
            if event.run_id != self.run_id:
                raise ValueError("event.run_id must match run.run_id")
        for metric in self.metrics:
            metric.validate()
            if metric.run_id != self.run_id:
                raise ValueError("metric.run_id must match run.run_id")
        for annotation in self.annotations:
            annotation.validate()
            if annotation.run_id != self.run_id:
                raise ValueError("annotation.run_id must match run.run_id")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        run = cls(
            run_id=data["run_id"],
            job=JobInfo(**data["job"]),
            workload=WorkloadInfo(**data["workload"]),
            events=[EventRecord.from_dict(item) for item in data.get("events", [])],
            metrics=[MetricRecord.from_dict(item) for item in data.get("metrics", [])],
            annotations=[AnnotationRecord.from_dict(item) for item in data.get("annotations", [])],
            schema_version=data.get("schema_version", "1.0.0"),
            collector_version=data.get("collector_version"),
            normalizer_version=data.get("normalizer_version"),
            source_coverage=data.get("source_coverage", {}),
            collection_errors=data.get("collection_errors", []),
            sampling_rate=data.get("sampling_rate", {}),
            dropped_event_counts=data.get("dropped_event_counts", {}),
        )
        run.validate()
        return run


def validate_run_dict(data: dict[str, Any]) -> None:
    RunRecord.from_dict(data)


def _require_non_empty(value: str | None, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


def _require_enum(value: str, allowed: tuple[str, ...], field_name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {allowed}")


def _require_non_negative_int(value: int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")


def _require_non_negative_optional_int(value: int | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be >= 0")
