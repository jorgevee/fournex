from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._native import HAS_NATIVE
from .analysis import summarize_run, summarize_run_with_steady_state
from .common_ir import RunRecord
from .sdk import get_local_events, get_runtime_config


def load_run_record(path: str | Path) -> RunRecord:
    """Load a serialized RunRecord JSON through the schema-version gate."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunRecord.from_dict(data)


def _resolve_events(events: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Resolve the event list to persist, guarding the native-backend trap.

    Under the native backend the Python event buffer is empty by design, so
    persisting it would silently write empty/misleading artifacts. Fail loudly
    instead — the native engine owns its own artifacts (fournex.flush()/
    shutdown()); callers wanting a Python-side trace must pass events explicitly.
    """
    if events is not None:
        return list(events)
    if HAS_NATIVE:
        raise RuntimeError(
            "persist_*: native telemetry backend is active; the Python event "
            "buffer is empty by design. Use the native engine's own finalization "
            "(fournex.flush()/shutdown()), or pass events= explicitly."
        )
    return get_local_events()


def persist_local_trace(
    events: list[dict[str, Any]] | None = None,
    *,
    output_path: str | None = None,
) -> str:
    runtime = get_runtime_config()
    resolved_events = _resolve_events(events)
    resolved_output = Path(output_path or runtime["raw_trace_path"])
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    with resolved_output.open("w", encoding="utf-8") as handle:
        for event in resolved_events:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")

    return str(resolved_output)


def persist_run_summary(
    events: list[dict[str, Any]] | None = None,
    *,
    output_path: str | None = None,
    summary: dict[str, Any] | None = None,
) -> str:
    runtime = get_runtime_config()
    resolved_events = _resolve_events(events)
    resolved_summary = summary if summary is not None else summarize_run(resolved_events)
    resolved_output = Path(output_path or runtime["derived_summary_path"])
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(json.dumps(resolved_summary, indent=2, sort_keys=True), encoding="utf-8")
    return str(resolved_output)


def persist_run_artifacts(events: list[dict[str, Any]] | None = None) -> dict[str, str]:
    runtime = get_runtime_config()
    resolved_events = _resolve_events(events)
    summary = summarize_run(resolved_events)
    raw_trace_path = persist_local_trace(resolved_events, output_path=runtime["raw_trace_path"])
    derived_summary_path = persist_run_summary(
        resolved_events,
        output_path=runtime["derived_summary_path"],
        summary=summary,
    )
    return {
        "raw_trace_path": raw_trace_path,
        "derived_summary_path": derived_summary_path,
    }


def persist_run_with_steady_state_summary(
    events: list[dict[str, Any]] | None = None,
    *,
    output_path: str | None = None,
    summary: dict[str, Any] | None = None,
    skip_first_n: int = 0,
    last_k: int | None = None,
) -> str:
    runtime = get_runtime_config()
    resolved_events = _resolve_events(events)
    resolved_summary = (
        summary
        if summary is not None
        else summarize_run_with_steady_state(
            resolved_events,
            skip_first_n=skip_first_n,
            last_k=last_k,
        )
    )
    resolved_output = Path(output_path or runtime["derived_summary_path"])
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(json.dumps(resolved_summary, indent=2, sort_keys=True), encoding="utf-8")
    return str(resolved_output)
