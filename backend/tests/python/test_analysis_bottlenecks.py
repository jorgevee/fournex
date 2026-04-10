import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import autopilot_telemetry as at

from analysis_bottleneck_golden_cases import (
    COPY_BOUND_EVENTS,
    INPUT_BOUND_EVENTS,
    LAUNCH_BOUND_EVENTS,
    MEMORY_PRESSURE_EVENTS,
    MIXED_SIGNAL_EVENTS,
    SHAPE_INSTABILITY_EVENTS,
    SPARSE_TELEMETRY_EVENTS,
    SYNC_BOUND_EVENTS,
)


def test_input_bound_golden_case() -> None:
    summary = at.summarize_run(INPUT_BOUND_EVENTS)
    assert summary["bottlenecks"][0]["label"] == "input_bound"
    assert summary["bottlenecks"][0]["evidence"]["avg_dataloader_fraction"] == 0.325
    assert summary["run_summary"]["dominant_stall_type"] == "input_bound"
    assert summary["diagnosis"]["primary_bottleneck"] == "input_bound"
    assert summary["diagnosis"]["secondary_bottlenecks"] == []
    assert summary["diagnosis"]["confidence"]["level"] == "medium"
    assert summary["diagnosis"]["recommendations"]


def test_copy_bound_golden_case() -> None:
    summary = at.summarize_run(COPY_BOUND_EVENTS)
    assert summary["bottlenecks"][0]["label"] == "copy_bound"
    assert summary["bottlenecks"][0]["evidence"]["steps_with_h2d"] == 2
    assert summary["run_summary"]["dominant_stall_type"] == "copy_bound"
    assert summary["diagnosis"]["primary_bottleneck"] == "copy_bound"
    assert summary["diagnosis"]["confidence"]["level"] == "medium"


def test_sync_bound_golden_case() -> None:
    summary = at.summarize_run(SYNC_BOUND_EVENTS)
    assert summary["bottlenecks"][0]["label"] == "sync_bound"
    assert summary["bottlenecks"][0]["evidence"]["avg_sync_fraction"] == 0.135
    assert summary["bottlenecks"][0]["evidence"]["steps_with_sync_wait"] == 2
    assert summary["run_summary"]["dominant_stall_type"] == "sync_bound"
    assert summary["diagnosis"]["primary_bottleneck"] == "sync_bound"
    assert summary["diagnosis"]["confidence"]["level"] == "medium"


def test_launch_bound_golden_case() -> None:
    summary = at.summarize_run(LAUNCH_BOUND_EVENTS)
    labels = [item["label"] for item in summary["bottlenecks"]]
    assert labels[0] == "underutilized_gpu"
    assert "launch_bound" in labels
    launch_result = next(item for item in summary["bottlenecks"] if item["label"] == "launch_bound")
    assert launch_result["evidence"]["profiler_windows_exported"] == 2
    assert summary["diagnosis"]["primary_bottleneck"] == "underutilized_gpu"
    assert summary["diagnosis"]["secondary_bottlenecks"] == ["launch_bound"]
    assert summary["diagnosis"]["why_not_others"]


def test_memory_pressure_golden_case() -> None:
    summary = at.summarize_run(MEMORY_PRESSURE_EVENTS)
    assert summary["bottlenecks"][0]["label"] == "memory_pressure"
    assert summary["run_summary"]["memory_pressure_peak_ratio"] == 0.95
    assert summary["diagnosis"]["primary_bottleneck"] == "memory_pressure"
    assert summary["diagnosis"]["confidence"]["level"] == "high"


def test_shape_instability_golden_case() -> None:
    summary = at.summarize_run(SHAPE_INSTABILITY_EVENTS)
    assert summary["bottlenecks"][0]["label"] == "shape_instability"
    assert summary["run_summary"]["shape_volatility_ratio"] == 1.0
    assert summary["bottlenecks"][0]["evidence"]["changed_steps"] == [2, 3]
    assert summary["diagnosis"]["primary_bottleneck"] == "shape_instability"
    assert summary["diagnosis"]["confidence"]["level"] == "high"
    assert summary["diagnosis"]["recommendations"]


def test_mixed_signal_case_ranks_multiple_bottlenecks() -> None:
    summary = at.summarize_run(MIXED_SIGNAL_EVENTS)
    labels = [item["label"] for item in summary["bottlenecks"]]
    assert labels[:3] == ["underutilized_gpu", "input_bound", "copy_bound"]
    assert summary["bottlenecks"][1]["score"] > summary["bottlenecks"][2]["score"]
    assert summary["diagnosis"]["primary_bottleneck"] == "underutilized_gpu"
    assert summary["diagnosis"]["secondary_bottlenecks"] == ["input_bound", "copy_bound"]
    assert summary["diagnosis"]["confidence"]["level"] == "high"


def test_sparse_telemetry_case_degrades_gracefully() -> None:
    per_step = at.derive_step_metrics(SPARSE_TELEMETRY_EVENTS)
    run_summary = at.derive_run_summary(SPARSE_TELEMETRY_EVENTS, per_step)
    bottlenecks = at.classify_bottlenecks(SPARSE_TELEMETRY_EVENTS, per_step, run_summary)
    diagnosis = at.build_diagnosis_result(bottlenecks, run_summary)

    assert len(per_step) == 2
    assert run_summary["average_gpu_utilization_pct"] == 0.0
    assert bottlenecks == []
    assert diagnosis["primary_bottleneck"] is None
    assert diagnosis["confidence"]["score"] == 0.0


def test_summarize_step_scope_can_classify_selected_steps() -> None:
    events = INPUT_BOUND_EVENTS + COPY_BOUND_EVENTS
    summary = at.summarize_step_scope(events, step_ids=[1, 2], scope_name="window_a")

    assert summary["scope"]["name"] == "window_a"
    assert summary["scope"]["step_ids"] == [1, 2]
    assert summary["step_count"] == 2
    assert summary["diagnosis"]["primary_bottleneck"] == "input_bound"


def test_summarize_step_scope_changes_diagnosis_for_later_steps() -> None:
    shifted_copy_events = []
    for event in COPY_BOUND_EVENTS:
        copied = dict(event)
        if "payload" in event:
            copied["payload"] = dict(event["payload"])
        if copied.get("step_id") is not None:
            copied["step_id"] = copied["step_id"] + 2
        shifted_copy_events.append(copied)

    events = INPUT_BOUND_EVENTS + shifted_copy_events
    summary = at.summarize_step_scope(events, step_ids=[3, 4], scope_name="window_b")

    assert summary["scope"]["name"] == "window_b"
    assert summary["scope"]["step_ids"] == [3, 4]
    assert summary["step_count"] == 2
    assert summary["diagnosis"]["primary_bottleneck"] == "copy_bound"


def test_select_steady_state_step_ids_skips_warmup_and_keeps_last_k() -> None:
    shifted_copy_events = []
    for event in COPY_BOUND_EVENTS:
        copied = dict(event)
        if "payload" in event:
            copied["payload"] = dict(event["payload"])
        if copied.get("step_id") is not None:
            copied["step_id"] = copied["step_id"] + 2
        shifted_copy_events.append(copied)

    per_step = at.derive_step_metrics(INPUT_BOUND_EVENTS + shifted_copy_events)
    step_ids = at.select_steady_state_step_ids(per_step, skip_first_n=2, last_k=2)

    assert step_ids == [3, 4]


def test_summarize_steady_state_can_skip_warmup_steps() -> None:
    shifted_copy_events = []
    for event in COPY_BOUND_EVENTS:
        copied = dict(event)
        if "payload" in event:
            copied["payload"] = dict(event["payload"])
        if copied.get("step_id") is not None:
            copied["step_id"] = copied["step_id"] + 2
        shifted_copy_events.append(copied)

    events = INPUT_BOUND_EVENTS + shifted_copy_events
    summary = at.summarize_steady_state(events, skip_first_n=2)

    assert summary["scope"]["name"] == "steady_state"
    assert summary["scope"]["step_ids"] == [3, 4]
    assert summary["step_count"] == 2
    assert summary["diagnosis"]["primary_bottleneck"] == "copy_bound"


def test_summarize_steady_state_can_limit_to_last_k_steps() -> None:
    shifted_copy_events = []
    for event in COPY_BOUND_EVENTS:
        copied = dict(event)
        if "payload" in event:
            copied["payload"] = dict(event["payload"])
        if copied.get("step_id") is not None:
            copied["step_id"] = copied["step_id"] + 2
        shifted_copy_events.append(copied)

    events = INPUT_BOUND_EVENTS + shifted_copy_events
    summary = at.summarize_steady_state(events, last_k=2)

    assert summary["scope"]["name"] == "steady_state"
    assert summary["scope"]["step_ids"] == [3, 4]
    assert summary["step_count"] == 2
    assert summary["diagnosis"]["primary_bottleneck"] == "copy_bound"


def test_summarize_run_with_steady_state_returns_both_scopes() -> None:
    shifted_copy_events = []
    for event in COPY_BOUND_EVENTS:
        copied = dict(event)
        if "payload" in event:
            copied["payload"] = dict(event["payload"])
        if copied.get("step_id") is not None:
            copied["step_id"] = copied["step_id"] + 2
        shifted_copy_events.append(copied)

    events = INPUT_BOUND_EVENTS + shifted_copy_events
    summary = at.summarize_run_with_steady_state(events, skip_first_n=2)

    assert summary["event_count"] == len(events)
    assert summary["step_count"] == 4
    assert summary["selector"] == {"policy": "explicit", "skip_first_n": 2, "last_k": None}
    assert summary["run"]["scope"]["name"] == "run"
    assert summary["run"]["scope"]["step_ids"] == [1, 2, 3, 4]
    assert summary["run"]["diagnosis"]["primary_bottleneck"] is None
    assert summary["run"]["diagnosis"]["dominant_stall_type"] == "input_bound"
    assert summary["steady_state"]["scope"]["name"] == "steady_state"
    assert summary["steady_state"]["scope"]["step_ids"] == [3, 4]
    assert summary["steady_state"]["diagnosis"]["primary_bottleneck"] == "copy_bound"
    assert summary["scope_comparison"]["diagnosis_changed"] is True
    assert summary["scope_comparison"]["run_primary_bottleneck"] is None
    assert summary["scope_comparison"]["steady_state_primary_bottleneck"] == "copy_bound"


def test_summarize_run_with_steady_state_uses_default_selector_policy() -> None:
    shifted_copy_events = []
    for event in COPY_BOUND_EVENTS:
        copied = dict(event)
        if "payload" in event:
            copied["payload"] = dict(event["payload"])
        if copied.get("step_id") is not None:
            copied["step_id"] = copied["step_id"] + 2
        shifted_copy_events.append(copied)

    events = INPUT_BOUND_EVENTS + shifted_copy_events
    summary = at.summarize_run_with_steady_state(events)

    assert summary["selector"] == {"policy": "default", "skip_first_n": 2, "last_k": None}
    assert summary["steady_state"]["scope"]["step_ids"] == [3, 4]
    assert summary["steady_state"]["diagnosis"]["primary_bottleneck"] == "copy_bound"
