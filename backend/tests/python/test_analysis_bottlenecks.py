import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as at

from analysis_bottleneck_golden_cases import (
    COPY_BOUND_EVENTS,
    INPUT_BOUND_EVENTS,
    LAUNCH_BOUND_EVENTS,
    MEMORY_PRESSURE_EVENTS,
    MIXED_SIGNAL_EVENTS,
    SHAPE_INSTABILITY_EVENTS,
    SPARSE_TELEMETRY_EVENTS,
    SYNC_BOUND_EVENTS,
    UNDERUTILIZED_GPU_EVENTS,
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


def test_sparse_telemetry_case_emits_insufficient_telemetry() -> None:
    per_step = at.derive_step_metrics(SPARSE_TELEMETRY_EVENTS)
    run_summary = at.derive_run_summary(SPARSE_TELEMETRY_EVENTS, per_step)
    bottlenecks = at.classify_bottlenecks(SPARSE_TELEMETRY_EVENTS, per_step, run_summary)
    diagnosis = at.build_diagnosis_result(bottlenecks, run_summary)

    assert len(per_step) == 2
    assert run_summary["average_gpu_utilization_pct"] == 0.0
    assert bottlenecks[0]["label"] == "insufficient_telemetry"
    assert bottlenecks[0]["score"] == 1.0
    assert bottlenecks[0]["evidence"]["step_count"] == 2
    assert diagnosis["primary_bottleneck"] == "insufficient_telemetry"
    assert diagnosis["confidence"]["level"] == "high"
    assert diagnosis["recommendations"]


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


def test_underutilized_gpu_standalone_golden_case() -> None:
    summary = at.summarize_run(UNDERUTILIZED_GPU_EVENTS)
    assert summary["bottlenecks"][0]["label"] == "underutilized_gpu"
    assert len(summary["bottlenecks"]) == 1
    assert summary["diagnosis"]["primary_bottleneck"] == "underutilized_gpu"
    assert summary["diagnosis"]["secondary_bottlenecks"] == []
    assert summary["diagnosis"]["confidence"]["level"] == "high"
    assert summary["diagnosis"]["recommendations"]


def test_classifier_version_present_in_diagnosis() -> None:
    summary = at.summarize_run(INPUT_BOUND_EVENTS)
    assert summary["diagnosis"]["classifier_version"] == "0.2.0"


def test_insufficient_telemetry_does_not_fire_when_timing_data_exists() -> None:
    summary = at.summarize_run(INPUT_BOUND_EVENTS)
    labels = [item["label"] for item in summary["bottlenecks"]]
    assert "insufficient_telemetry" not in labels


def test_worst_steps_input_bound_ordered_by_dataloader_fraction() -> None:
    # Step 1: dataloader=35/100=0.35, Step 2: dataloader=30/100=0.30
    summary = at.summarize_run(INPUT_BOUND_EVENTS)
    worst = summary["bottlenecks"][0]["worst_steps"]
    assert worst[0] == {"step_id": 1, "value": 0.35}
    assert worst[1] == {"step_id": 2, "value": 0.30}


def test_worst_steps_copy_bound_ordered_by_h2d_fraction() -> None:
    # Step 1: h2d=20/100=0.20, Step 2: h2d=18/100=0.18
    summary = at.summarize_run(COPY_BOUND_EVENTS)
    worst = summary["bottlenecks"][0]["worst_steps"]
    assert worst[0] == {"step_id": 1, "value": 0.20}
    assert worst[1] == {"step_id": 2, "value": 0.18}


def test_worst_steps_sync_bound_ordered_by_sync_fraction() -> None:
    # Step 1: sync=15/100=0.15, Step 2: sync=12/100=0.12
    summary = at.summarize_run(SYNC_BOUND_EVENTS)
    worst = summary["bottlenecks"][0]["worst_steps"]
    assert worst[0] == {"step_id": 1, "value": 0.15}
    assert worst[1] == {"step_id": 2, "value": 0.12}


def test_worst_steps_shape_instability_lists_changed_steps() -> None:
    # Steps 2 and 3 change shape; step 1 does not
    summary = at.summarize_run(SHAPE_INSTABILITY_EVENTS)
    worst = summary["bottlenecks"][0]["worst_steps"]
    step_ids = [entry["step_id"] for entry in worst]
    assert step_ids == [2, 3]
    assert all(entry["value"] == 1.0 for entry in worst)


def test_worst_steps_memory_pressure_is_empty() -> None:
    summary = at.summarize_run(MEMORY_PRESSURE_EVENTS)
    assert summary["bottlenecks"][0]["label"] == "memory_pressure"
    assert summary["bottlenecks"][0]["worst_steps"] == []


def test_worst_steps_capped_at_three() -> None:
    # Build a 5-step run; worst_steps should return at most 3
    events = []
    for i in range(1, 6):
        events += [
            {"event_type": "step_start", "step_id": i, "payload": {"step_kind": "train"}},
            {"event_type": "dataloader_span", "step_id": i, "duration_ns": 20 + i, "payload": {"stage": "next"}},
            {"event_type": "phase_span", "step_id": i, "duration_ns": 30, "payload": {"phase_name": "forward"}},
            {"event_type": "step_end", "step_id": i, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
        ]
    summary = at.summarize_run(events)
    worst = summary["bottlenecks"][0]["worst_steps"]
    assert len(worst) <= 3
    # Values should be descending
    values = [entry["value"] for entry in worst]
    assert values == sorted(values, reverse=True)
