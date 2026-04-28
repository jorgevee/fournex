import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from autopilot_telemetry.autopilot.benchmark import BenchmarkWindow, apply_benchmark_window
from autopilot_telemetry.autopilot.guards import extract_metrics_from_summary


def _summary_with_steps():
    return {
        "run": {
            "scope": {"name": "run", "step_ids": [1, 2, 3, 4]},
            "step_count": 4,
            "per_step": [
                {
                    "step_id": 1,
                    "status": "ok",
                    "step_wall_time_ns": 1_000_000_000,
                    "dataloader_wait_time_ns": 500_000_000,
                    "h2d_copy_time_ns": 0,
                    "sync_wait_time_ns": 0,
                    "shape_changed": False,
                    "profiler_windows_exported": 0,
                },
                {
                    "step_id": 2,
                    "status": "ok",
                    "step_wall_time_ns": 500_000_000,
                    "dataloader_wait_time_ns": 0,
                    "h2d_copy_time_ns": 100_000_000,
                    "sync_wait_time_ns": 0,
                    "shape_changed": False,
                    "profiler_windows_exported": 0,
                },
                {
                    "step_id": 3,
                    "status": "ok",
                    "step_wall_time_ns": 250_000_000,
                    "dataloader_wait_time_ns": 0,
                    "h2d_copy_time_ns": 100_000_000,
                    "sync_wait_time_ns": 0,
                    "shape_changed": False,
                    "profiler_windows_exported": 0,
                },
                {
                    "step_id": 4,
                    "status": "ok",
                    "step_wall_time_ns": 250_000_000,
                    "dataloader_wait_time_ns": 0,
                    "h2d_copy_time_ns": 100_000_000,
                    "sync_wait_time_ns": 0,
                    "shape_changed": False,
                    "profiler_windows_exported": 0,
                },
            ],
            "run_summary": {
                "throughput_steps_per_sec": 2.0,
                "average_gpu_utilization_pct": 44.0,
                "average_memory_utilization_pct": 33.0,
                "memory_pressure_peak_ratio": 0.40,
                "utilization_instability_pct": 0.0,
                "step_time_avg_ns": 500_000_000,
                "step_time_max_ns": 1_000_000_000,
                "dominant_stall_type": "input_bound",
            },
        }
    }


def test_benchmark_window_env_and_validation():
    window = BenchmarkWindow(warmup_steps=2, measurement_steps=5, repeat_count=3, timeout_s=90)

    assert window.max_steps == 7
    assert window.env_vars() == {
        "FRX_TUNE_WARMUP_STEPS": "2",
        "FRX_TUNE_MEASURE_STEPS": "5",
        "FRX_TUNE_MAX_STEPS": "7",
        "FRX_TUNE_REPEAT_COUNT": "3",
    }


def test_benchmark_window_rejects_invalid_measurement_steps():
    try:
        BenchmarkWindow(measurement_steps=0)
    except ValueError as exc:
        assert "measurement_steps" in str(exc)
    else:
        raise AssertionError("expected invalid measurement_steps to raise")


def test_apply_benchmark_window_adds_measurement_scope_and_metrics_prefer_it():
    summary = apply_benchmark_window(
        _summary_with_steps(),
        BenchmarkWindow(warmup_steps=1, measurement_steps=2, timeout_s=30),
    )

    assert summary["benchmark_window"]["warmup_steps"] == 1
    assert summary["measurement_window"]["scope"]["step_ids"] == [2, 3]

    metrics = extract_metrics_from_summary(summary)
    assert metrics["step_count"] == 2
    assert metrics["throughput_steps_per_sec"] == 2 / 0.75
    assert metrics["avg_step_time_ms"] == 375.0
    assert metrics["dominant_stall"] == "copy_bound"


def test_benchmark_window_metadata_is_json_serializable():
    window = BenchmarkWindow(warmup_steps=1, measurement_steps=2, repeat_count=1, timeout_s=30)
    encoded = json.dumps(window.to_dict())

    assert json.loads(encoded)["measurement_steps"] == 2
