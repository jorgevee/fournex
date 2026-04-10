import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import autopilot_telemetry as at

from analysis_bottleneck_golden_cases import COPY_BOUND_EVENTS, INPUT_BOUND_EVENTS, MIXED_SIGNAL_EVENTS


OUTPUT_ROOT = ROOT / "traces" / "test_outputs"


def test_persist_run_summary_includes_diagnosis() -> None:
    output_path = OUTPUT_ROOT / "summary_input_bound.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written_path = at.persist_run_summary(INPUT_BOUND_EVENTS, output_path=str(output_path))

    payload = json.loads(Path(written_path).read_text(encoding="utf-8"))
    assert payload["bottlenecks"][0]["label"] == "input_bound"
    assert payload["diagnosis"]["primary_bottleneck"] == "input_bound"
    assert payload["diagnosis"]["secondary_bottlenecks"] == []
    assert payload["diagnosis"]["confidence"]["level"] == "medium"


def test_persist_run_summary_keeps_mixed_signal_diagnosis_shape() -> None:
    output_path = OUTPUT_ROOT / "summary_mixed_signal.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written_path = at.persist_run_summary(MIXED_SIGNAL_EVENTS, output_path=str(output_path))

    payload = json.loads(Path(written_path).read_text(encoding="utf-8"))
    assert payload["diagnosis"]["primary_bottleneck"] == "underutilized_gpu"
    assert payload["diagnosis"]["secondary_bottlenecks"] == ["input_bound", "copy_bound"]
    assert payload["diagnosis"]["why"]
    assert payload["diagnosis"]["why_not_others"]
    assert payload["diagnosis"]["recommendations"]


def test_persist_run_with_steady_state_summary_includes_both_scopes() -> None:
    shifted_copy_events = []
    for event in COPY_BOUND_EVENTS:
        copied = dict(event)
        if "payload" in event:
            copied["payload"] = dict(event["payload"])
        if copied.get("step_id") is not None:
            copied["step_id"] = copied["step_id"] + 2
        shifted_copy_events.append(copied)

    output_path = OUTPUT_ROOT / "summary_run_with_steady_state.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written_path = at.persist_run_with_steady_state_summary(
        INPUT_BOUND_EVENTS + shifted_copy_events,
        output_path=str(output_path),
        skip_first_n=2,
    )

    payload = json.loads(Path(written_path).read_text(encoding="utf-8"))
    assert payload["selector"] == {"policy": "explicit", "skip_first_n": 2, "last_k": None}
    assert payload["run"]["scope"]["name"] == "run"
    assert payload["run"]["diagnosis"]["primary_bottleneck"] is None
    assert payload["run"]["diagnosis"]["dominant_stall_type"] == "input_bound"
    assert payload["steady_state"]["scope"]["name"] == "steady_state"
    assert payload["steady_state"]["scope"]["step_ids"] == [3, 4]
    assert payload["steady_state"]["diagnosis"]["primary_bottleneck"] == "copy_bound"
    assert payload["scope_comparison"]["diagnosis_changed"] is True
