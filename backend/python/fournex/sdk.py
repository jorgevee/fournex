from __future__ import annotations

import atexit
import os
import threading
import time
import uuid
from typing import Any

from ._native import HAS_NATIVE, native

SCHEMA_VERSION = "0.1.0"

EVENT_TYPES = (
    "gpu_sample",
    "step_start",
    "step_end",
    "phase_span",
    "dataloader_span",
    "memcpy_span",
    "shape_snapshot",
    "sync_wait",
    "profiler_window",
    "system_info",
    "warning_annotation",
)

EVENT_SOURCES = ("python_sdk", "native_engine")
EVENT_LEVELS = ("debug", "info", "warning", "error")

_runtime_config: dict[str, Any] = {
    "job_name": "unknown-job",
    "output_dir": os.path.join(".", "traces"),
    "raw_output_dir": os.path.join(".", "traces", "raw"),
    "derived_output_dir": os.path.join(".", "traces", "derived"),
    "output_path": os.path.join(".", "traces", "raw", "trace.jsonl"),
    "raw_trace_path": os.path.join(".", "traces", "raw", "trace.jsonl"),
    "derived_summary_path": os.path.join(".", "traces", "derived", "summary.json"),
    "sample_interval_ms": 1000,
    "run_id": "unknown-run",
    "enable_cupti": False,
    "cupti_debug_mode": False,
}
_local_events: list[dict[str, Any]] = []
_auto_persist_registered = False
_auto_persist_done = False


def init(**kwargs: Any) -> dict[str, Any]:
    job_name = kwargs.get("job_name") or os.environ.get("FRX_JOB_NAME", "unknown-job")
    output_dir = kwargs.get("output_dir") or os.environ.get("FRX_OUTPUT_DIR", os.path.join(".", "traces"))
    raw_output_dir = kwargs.get("raw_output_dir", os.path.join(output_dir, "raw"))
    derived_output_dir = kwargs.get("derived_output_dir", os.path.join(output_dir, "derived"))
    sample_interval_ms = int(
        kwargs.get("sample_interval_ms") or os.environ.get("FRX_SAMPLE_INTERVAL_MS", 1000)
    )
    run_id = kwargs.get("run_id") or os.environ.get("FRX_RUN_ID", f"run-{uuid.uuid4().hex[:12]}")
    raw_trace_path = (
        kwargs.get("raw_trace_path")
        or os.environ.get("FRX_RAW_TRACE_PATH")
        or os.path.join(raw_output_dir, f"{run_id}.jsonl")
    )
    derived_summary_path = kwargs.get(
        "derived_summary_path",
        os.environ.get("FRX_DERIVED_SUMMARY_PATH", os.path.join(derived_output_dir, f"{run_id}_summary.json")),
    )
    output_path = kwargs.get("output_path", raw_trace_path)
    enable_cupti = bool(kwargs.get("enable_cupti", False))
    cupti_debug_mode = bool(kwargs.get("cupti_debug_mode", False))

    _runtime_config.update(
        {
            "job_name": job_name,
            "output_dir": output_dir,
            "raw_output_dir": raw_output_dir,
            "derived_output_dir": derived_output_dir,
            "output_path": output_path,
            "raw_trace_path": raw_trace_path,
            "derived_summary_path": derived_summary_path,
            "sample_interval_ms": sample_interval_ms,
            "run_id": run_id,
            "enable_cupti": enable_cupti,
            "cupti_debug_mode": cupti_debug_mode,
        }
    )

    if HAS_NATIVE:
        native.init(job_name=job_name, output_path=raw_trace_path,
                    sample_interval_ms=sample_interval_ms, run_id=run_id,
                    enable_cupti=enable_cupti, cupti_debug_mode=cupti_debug_mode)
    else:
        _local_events.clear()

    if os.environ.get("FRX_AUTO_PERSIST") == "1":
        _register_auto_persist()

    return dict(_runtime_config)


def flush() -> None:
    if HAS_NATIVE:
        native.flush()
    return None


def shutdown() -> None:
    if HAS_NATIVE:
        native.shutdown()
    return None


def make_event(
    *,
    event_id: str,
    timestamp_ns: int,
    pid: int,
    tid: int,
    job_id: str,
    run_id: str,
    event_type: str,
    event_source: str,
    level: str = "info",
    payload: dict[str, Any] | None = None,
    gpu_id: int | None = None,
    step_id: int | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    duration_ns: int | None = None,
) -> dict[str, Any]:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unsupported event_type: {event_type}")
    if event_source not in EVENT_SOURCES:
        raise ValueError(f"unsupported event_source: {event_source}")
    if level not in EVENT_LEVELS:
        raise ValueError(f"unsupported level: {level}")

    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "timestamp_ns": timestamp_ns,
        "pid": pid,
        "tid": tid,
        "job_id": job_id,
        "run_id": run_id,
        "event_type": event_type,
        "event_source": event_source,
        "gpu_id": gpu_id,
        "step_id": step_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "duration_ns": duration_ns,
        "level": level,
        "payload": payload or {},
    }


def emit_event(event: dict[str, Any]) -> dict[str, Any]:
    if HAS_NATIVE:
        native.emit_event(event)
    else:
        _local_events.append(event)
    return event


def begin_span(event: dict[str, Any]) -> dict[str, Any]:
    if HAS_NATIVE:
        native.begin_span(event)
    else:
        _local_events.append(event)
    return event


def end_span(event: dict[str, Any]) -> dict[str, Any]:
    if HAS_NATIVE:
        native.end_span(event)
    else:
        _local_events.append(event)
    return event


def build_runtime_event(
    *,
    event_type: str,
    payload: dict[str, Any] | None = None,
    level: str = "info",
    gpu_id: int | None = None,
    step_id: int | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    duration_ns: int | None = None,
) -> dict[str, Any]:
    return make_event(
        event_id=f"evt-{uuid.uuid4().hex}",
        timestamp_ns=time.perf_counter_ns(),
        pid=os.getpid(),
        tid=threading.get_native_id(),
        job_id=_runtime_config["job_name"],
        run_id=_runtime_config["run_id"],
        event_type=event_type,
        event_source="python_sdk",
        level=level,
        payload=payload or {},
        gpu_id=gpu_id,
        step_id=step_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        duration_ns=duration_ns,
    )


def get_local_events() -> list[dict[str, Any]]:
    return list(_local_events)


def clear_local_events() -> None:
    _local_events.clear()


def get_runtime_config() -> dict[str, Any]:
    return dict(_runtime_config)


def _register_auto_persist() -> None:
    global _auto_persist_registered
    if _auto_persist_registered:
        return
    atexit.register(_auto_persist_artifacts)
    _auto_persist_registered = True


def _auto_persist_artifacts() -> None:
    global _auto_persist_done
    if _auto_persist_done:
        return
    _auto_persist_done = True
    try:
        from .storage import persist_run_artifacts

        persist_run_artifacts()
    except Exception:
        # Auto-persist must never mask the workload's exit status.
        return
