import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as at

from analysis_bottleneck_golden_cases import (
    COPY_BOUND_EVENTS,
    INPUT_BOUND_EVENTS,
    LAUNCH_BOUND_EVENTS,
    LAUNCH_BOUND_TINY_KERNEL_EVENTS,
    MEMORY_PRESSURE_EVENTS,
    MIXED_SIGNAL_EVENTS,
    SHAPE_INSTABILITY_EVENTS,
    SPARSE_TELEMETRY_EVENTS,
    SYNC_BOUND_EVENTS,
    UNDERUTILIZED_GPU_EVENTS,
)


GOLDEN_EVENT_FIXTURES = {
    "input_bound": INPUT_BOUND_EVENTS,
    "copy_bound": COPY_BOUND_EVENTS,
    "sync_bound": SYNC_BOUND_EVENTS,
    "launch_bound": LAUNCH_BOUND_EVENTS,
    "launch_bound_tiny_kernel": LAUNCH_BOUND_TINY_KERNEL_EVENTS,
    "memory_pressure": MEMORY_PRESSURE_EVENTS,
    "shape_instability": SHAPE_INSTABILITY_EVENTS,
    "mixed_signal": MIXED_SIGNAL_EVENTS,
    "sparse_telemetry": SPARSE_TELEMETRY_EVENTS,
    "underutilized_gpu": UNDERUTILIZED_GPU_EVENTS,
}


def test_golden_fixtures_have_multiple_complete_steps() -> None:
    for name, events in GOLDEN_EVENT_FIXTURES.items():
        started = {
            event["step_id"]
            for event in events
            if event.get("event_type") == "step_start"
        }
        completed = {
            event["step_id"]
            for event in events
            if event.get("event_type") == "step_end"
        }

        assert len(completed) >= 2, f"{name} must cover averaging across multiple steps"
        assert completed == started, f"{name} has mismatched step_start/step_end boundaries"


def _single_signal_steps(
    signal: str,
    duration_ns: int,
    *,
    steps: int = 2,
    step_wall_ns: int = 1000,
) -> list[dict]:
    events: list[dict] = [
        {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 60, "utilization_mem_pct": 35, "memory_used_bytes": 40, "memory_total_bytes": 100}},
    ]
    for step_id in range(1, steps + 1):
        events.append({"event_type": "step_start", "step_id": step_id, "payload": {"step_kind": "train"}})
        if signal == "input":
            events.append({"event_type": "dataloader_span", "step_id": step_id, "duration_ns": duration_ns, "payload": {"stage": "next"}})
        elif signal == "copy":
            events.append({"event_type": "memcpy_span", "step_id": step_id, "duration_ns": duration_ns, "payload": {"copy_kind": "h2d"}})
        elif signal == "sync":
            events.append({"event_type": "sync_wait", "step_id": step_id, "duration_ns": duration_ns, "payload": {"wait_kind": "device_sync"}})
        events += [
            {"event_type": "phase_span", "step_id": step_id, "duration_ns": 300, "payload": {"phase_name": "forward"}},
            {"event_type": "phase_span", "step_id": step_id, "duration_ns": 250, "payload": {"phase_name": "backward"}},
            {"event_type": "step_end", "step_id": step_id, "duration_ns": step_wall_ns, "payload": {"step_kind": "train", "status": "ok"}},
        ]
    return events


def _memory_pressure_steps(used_bytes: int, total_bytes: int = 100) -> list[dict]:
    return [
        {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 65, "utilization_mem_pct": 80, "memory_used_bytes": used_bytes, "memory_total_bytes": total_bytes}},
        {"event_type": "step_start", "step_id": 1, "payload": {"step_kind": "train"}},
        {"event_type": "phase_span", "step_id": 1, "duration_ns": 50, "payload": {"phase_name": "forward"}},
        {"event_type": "step_end", "step_id": 1, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
        {"event_type": "step_start", "step_id": 2, "payload": {"step_kind": "train"}},
        {"event_type": "phase_span", "step_id": 2, "duration_ns": 50, "payload": {"phase_name": "forward"}},
        {"event_type": "step_end", "step_id": 2, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
    ]


def _shape_sequence_steps(sequence_lengths: list[int]) -> list[dict]:
    events = [
        {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 65, "utilization_mem_pct": 40, "memory_used_bytes": 40, "memory_total_bytes": 100}},
    ]
    for step_id, sequence_length in enumerate(sequence_lengths, start=1):
        events += [
            {"event_type": "step_start", "step_id": step_id, "payload": {"step_kind": "train"}},
            {"event_type": "shape_snapshot", "step_id": step_id, "payload": {"batch_size": 16, "sequence_length": sequence_length, "shapes": {"x": [16, sequence_length]}}},
            {"event_type": "phase_span", "step_id": step_id, "duration_ns": 50, "payload": {"phase_name": "forward"}},
            {"event_type": "step_end", "step_id": step_id, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
        ]
    return events


def _labels(events: list[dict]) -> list[str]:
    return [item["label"] for item in at.summarize_run(events)["bottlenecks"]]


def test_classifier_threshold_boundaries_are_inclusive() -> None:
    shape_at_threshold = _shape_sequence_steps([128, 128, 128, 128, 256, 256, 256, 384, 384, 512, 512])

    assert "input_bound" in _labels(_single_signal_steps("input", 200))
    assert "copy_bound" in _labels(_single_signal_steps("copy", 150))
    assert "sync_bound" in _labels(_single_signal_steps("sync", 100))
    assert "memory_pressure" in _labels(_memory_pressure_steps(90))
    assert "shape_instability" in _labels(shape_at_threshold)
    assert at.summarize_run(shape_at_threshold)["run_summary"]["shape_volatility_ratio"] == 0.3


def test_classifier_threshold_boundaries_do_not_fire_just_below() -> None:
    shape_below_threshold = _shape_sequence_steps([128, 128, 128, 128, 128, 256, 256, 256, 384, 384, 384])

    assert "input_bound" not in _labels(_single_signal_steps("input", 199))
    assert "copy_bound" not in _labels(_single_signal_steps("copy", 149))
    assert "sync_bound" not in _labels(_single_signal_steps("sync", 99))
    assert "memory_pressure" not in _labels(_memory_pressure_steps(89))
    assert "shape_instability" not in _labels(shape_below_threshold)
    assert at.summarize_run(shape_below_threshold)["run_summary"]["shape_volatility_ratio"] == 0.2


def test_input_bound_confidence_calibration() -> None:
    high_confidence = at.summarize_run(_single_signal_steps("input", 500))
    medium_confidence = at.summarize_run(_single_signal_steps("input", 200))

    assert high_confidence["diagnosis"]["primary_bottleneck"] == "input_bound"
    assert high_confidence["diagnosis"]["confidence"]["level"] == "high"
    assert medium_confidence["diagnosis"]["primary_bottleneck"] == "input_bound"
    assert medium_confidence["diagnosis"]["confidence"]["level"] == "medium"


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


def test_launch_bound_tiny_kernel_golden_case() -> None:
    summary = at.summarize_run(LAUNCH_BOUND_TINY_KERNEL_EVENTS)
    labels = [item["label"] for item in summary["bottlenecks"]]
    launch_result = next(item for item in summary["bottlenecks"] if item["label"] == "launch_bound")

    assert labels[0] == "underutilized_gpu"
    assert "launch_bound" in labels
    assert summary["run_summary"]["profiler_windows_exported"] == 2
    assert summary["run_summary"]["kernel_count_per_step"] == 200
    assert summary["run_summary"]["median_cuda_kernel_duration_us"] == 7.5
    assert summary["run_summary"]["small_kernel_fraction"] == 0.84
    assert launch_result["evidence"]["kernel_count_per_step"] == 200
    assert launch_result["evidence"]["median_cuda_kernel_duration_us"] == 7.5
    assert launch_result["evidence"]["small_kernel_fraction"] == 0.84
    assert summary["diagnosis"]["secondary_bottlenecks"] == ["launch_bound"]


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
