from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analysis import summarize_run
from .sdk import get_local_events, get_runtime_config


def persist_local_trace(
    events: list[dict[str, Any]] | None = None,
    *,
    output_path: str | None = None,
) -> str:
    runtime = get_runtime_config()
    resolved_events = list(events) if events is not None else get_local_events()
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
    resolved_events = list(events) if events is not None else get_local_events()
    resolved_summary = summary if summary is not None else summarize_run(resolved_events)
    resolved_output = Path(output_path or runtime["derived_summary_path"])
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(json.dumps(resolved_summary, indent=2, sort_keys=True), encoding="utf-8")
    return str(resolved_output)


def persist_run_artifacts(events: list[dict[str, Any]] | None = None) -> dict[str, str]:
    runtime = get_runtime_config()
    resolved_events = list(events) if events is not None else get_local_events()
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
