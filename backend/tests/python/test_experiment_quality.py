import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from autopilot_telemetry.autopilot.actions import TrialResult
from autopilot_telemetry.autopilot.quality import (
    check_quality_regression,
    extract_quality_metrics,
)


def _trial(name: str, quality_metrics: dict):
    return TrialResult(
        config_id=name,
        label=name,
        exit_code=0,
        throughput_steps_per_sec=10.0,
        avg_gpu_utilization_pct=50.0,
        avg_step_time_ms=100.0,
        peak_memory_ratio=0.25,
        dominant_stall="none",
        step_count=4,
        passed_guards=True,
        guard_failures=[],
        quality_metrics=quality_metrics,
    )


def test_extract_quality_metrics_from_measurement_window_losses():
    summary = {
        "measurement_window": {
            "per_step": [
                {"step_id": 2, "loss": 1.0},
                {"step_id": 3, "loss": 0.8},
                {"step_id": 4, "loss": 0.7},
            ],
            "run_summary": {},
        }
    }

    metrics = extract_quality_metrics(summary)

    assert metrics["loss_count"] == 3
    assert metrics["initial_loss"] == 1.0
    assert metrics["final_loss"] == 0.7
    assert metrics["loss_slope"] == -0.30000000000000004


def test_quality_gate_rejects_final_loss_regression():
    baseline = _trial("baseline", {"final_loss": 1.0})
    trial = _trial("trial", {"final_loss": 1.10})

    passed, failures = check_quality_regression(baseline, trial)

    assert not passed
    assert any("final loss" in failure for failure in failures)


def test_quality_gate_rejects_nonfinite_loss():
    baseline = _trial("baseline", {"final_loss": 1.0})
    trial = _trial("trial", {"nan_count": 1, "final_loss": 0.8})

    passed, failures = check_quality_regression(baseline, trial)

    assert not passed
    assert any("NaN" in failure for failure in failures)


def test_quality_gate_rejects_output_drift():
    baseline = _trial("baseline", {})
    trial = _trial("trial", {"max_output_abs_diff": 0.02})

    passed, failures = check_quality_regression(baseline, trial)

    assert not passed
    assert any("output absolute difference" in failure for failure in failures)
