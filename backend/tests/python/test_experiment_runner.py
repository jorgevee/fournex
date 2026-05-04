import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex.autopilot.actions import PromotionThresholds, TrialConfig
from fournex.autopilot.local_executor import LocalTrialExecutor
from fournex.autopilot.runner import ExperimentRunner


SUMMARY_WRITER = r"""
import json
import os
from pathlib import Path

throughput = float(os.environ.get("FRX_TEST_THROUGHPUT", "12.5"))
summary_path = Path(os.environ["FRX_DERIVED_SUMMARY_PATH"])
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps({
    "steady_state": {
        "step_count": 5,
        "run_summary": {
            "throughput_steps_per_sec": throughput,
            "average_gpu_utilization_pct": 51.0,
            "step_time_avg_ns": 80000000,
            "memory_pressure_peak_ratio": 0.25,
            "dominant_stall_type": "none"
        }
    }
}), encoding="utf-8")
print("trial stdout")
"""

QUALITY_WRITER = r"""
import json
import os
from pathlib import Path

is_trial = os.environ.get("FRX_PIN_MEMORY") == "true"
throughput = 20.0 if is_trial else 10.0
final_loss = 1.20 if is_trial else 1.0
summary_path = Path(os.environ["FRX_DERIVED_SUMMARY_PATH"])
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps({
    "steady_state": {
        "step_count": 5,
        "run_summary": {
            "throughput_steps_per_sec": throughput,
            "average_gpu_utilization_pct": 51.0,
            "step_time_avg_ns": 80000000,
            "memory_pressure_peak_ratio": 0.25,
            "dominant_stall_type": "none",
            "final_loss": final_loss,
            "nan_count": 0,
            "inf_count": 0
        }
    }
}), encoding="utf-8")
"""

REPEAT_NOISE_WRITER = r"""
import json
import os
from pathlib import Path

out_dir = Path(os.environ["FRX_OUTPUT_DIR"])
repeat_name = out_dir.name
repeat_index = int(repeat_name.split("_")[-1]) if repeat_name.startswith("repeat_") else 1
is_trial = os.environ.get("FRX_PIN_MEMORY") == "true"
if is_trial:
    throughput = 114.0 if repeat_index == 1 else 116.0
else:
    throughput = 100.0 if repeat_index == 1 else 120.0

summary_path = Path(os.environ["FRX_DERIVED_SUMMARY_PATH"])
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps({
    "steady_state": {
        "step_count": 5,
        "run_summary": {
            "throughput_steps_per_sec": throughput,
            "average_gpu_utilization_pct": 51.0,
            "step_time_avg_ns": int(1_000_000_000 / throughput),
            "memory_pressure_peak_ratio": 0.25,
            "dominant_stall_type": "none"
        }
    }
}), encoding="utf-8")
"""

BATCH_SIZE_TRACE_WRITER = r"""
import json
import os
from pathlib import Path

batch_size = int(os.environ.get("FRX_BATCH_SIZE", "32"))
throughput = 14.0 if batch_size > 32 else 10.0
step_ns = int(1_000_000_000 / throughput)
summary_path = Path(os.environ["FRX_DERIVED_SUMMARY_PATH"])
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps({
    "steady_state": {
        "step_count": 5,
        "per_step": [
            {
                "step_id": i,
                "status": "ok",
                "step_wall_time_ns": step_ns,
                "batch_size": batch_size,
                "shape_changed": False,
                "profiler_windows_exported": 0
            }
            for i in range(1, 6)
        ],
        "run_summary": {
            "throughput_steps_per_sec": throughput,
            "average_gpu_utilization_pct": 40.0,
            "step_time_avg_ns": step_ns,
            "memory_pressure_peak_ratio": 0.25,
            "dominant_stall_type": "compute_bound"
        }
    }
}), encoding="utf-8")
"""

RACE_WRITER = r"""
import json
import os
from pathlib import Path

nw = os.environ.get("FRX_NUM_WORKERS")
pin = os.environ.get("FRX_PIN_MEMORY")
key = (nw, pin)
throughput_by_config = {
    (None, None): 100.0,
    ("0", "true"): 90.0,
    ("2", "true"): 130.0,
    ("2", "false"): 110.0,
    ("4", "true"): 150.0,
    ("4", "false"): 95.0,
}
throughput = throughput_by_config.get(key, 100.0)
summary_path = Path(os.environ["FRX_DERIVED_SUMMARY_PATH"])
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps({
    "steady_state": {
        "step_count": 5,
        "run_summary": {
            "throughput_steps_per_sec": throughput,
            "average_gpu_utilization_pct": 55.0,
            "step_time_avg_ns": int(1_000_000_000 / throughput),
            "memory_pressure_peak_ratio": 0.25,
            "dominant_stall_type": "input_bound"
        }
    }
}), encoding="utf-8")
"""


def _workspace_tmp(name: str) -> Path:
    path = ROOT / "traces" / "cli_test_runs" / "experiment_runner" / f"{name}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_local_trial_executor_writes_phase1_artifacts():
    tmp_path = _workspace_tmp("local-executor")
    executor = LocalTrialExecutor(
        workload_command=[sys.executable, "-c", SUMMARY_WRITER],
        job_name="unit",
        time_budget_s=5,
        thresholds=PromotionThresholds(require_sufficient_steps=3),
        verbose=False,
    )

    result = executor.run(
        TrialConfig(
            name="trial_001",
            label="candidate",
            patch={"num_workers": 8},
            env={"FRX_TEST_THROUGHPUT": "18.0"},
            risk_level="low",
        ),
        tmp_path / "trial_001",
    )

    trial_dir = tmp_path / "trial_001"
    metrics = json.loads((trial_dir / "metrics.json").read_text(encoding="utf-8"))

    assert result.is_viable
    assert result.throughput_steps_per_sec == 18.0
    assert (trial_dir / "config.yaml").exists()
    assert json.loads((trial_dir / "benchmark_window.json").read_text(encoding="utf-8")) == {
        "warmup_steps": 5,
        "measurement_steps": 20,
        "repeat_count": 1,
        "timeout_s": 5,
    }
    assert (trial_dir / "stdout.log").read_text(encoding="utf-8") == "trial stdout\n"
    assert (trial_dir / "stderr.log").read_text(encoding="utf-8") == ""
    assert metrics["config_id"] == "trial_001"
    assert metrics["artifact_paths"]["metrics"] == "metrics.json"


def test_experiment_runner_persists_json_and_markdown_reports():
    tmp_path = _workspace_tmp("runner")
    runner = ExperimentRunner(
        workload_command=[sys.executable, "-c", SUMMARY_WRITER],
        job_name="unit-runner",
        out_dir=str(tmp_path),
        max_trials=1,
        time_budget_s=5,
        thresholds=PromotionThresholds(min_speedup=0.01, require_sufficient_steps=3),
        environment={"cpu_count": 2},
        verbose=False,
    )

    report = runner.run()

    tune_dirs = list(tmp_path.glob("tune-*"))
    assert len(tune_dirs) == 1
    tune_dir = tune_dirs[0]
    assert (tune_dir / "baseline" / "metrics.json").exists()
    assert (tune_dir / "baseline" / "benchmark_window.json").exists()
    assert (tune_dir / "dl_nw0_pin1" / "metrics.json").exists()
    assert (tune_dir / "autopilot_report.json").exists()
    assert (tune_dir / "report.md").exists()
    assert report.total_trials == 1


def test_experiment_runner_records_unsafe_candidate_without_executing_it():
    tmp_path = _workspace_tmp("runner-safety")
    runner = ExperimentRunner(
        workload_command=[sys.executable, "-c", SUMMARY_WRITER],
        job_name="unit-runner-safety",
        out_dir=str(tmp_path),
        max_trials=1,
        safe_only=False,
        time_budget_s=5,
        bottleneck_diagnosis="launch_bound",
        environment={"torch_compile_supported": False},
        verbose=False,
    )

    report = runner.run()

    tune_dir = next(tmp_path.glob("tune-*"))
    skipped = report.trials[0]
    assert skipped.config_id == "compile_default"
    assert skipped.exit_code == -3
    assert skipped.throughput_steps_per_sec == 0.0
    assert any("pre-run safety validation" in reason for reason in skipped.guard_failures)
    assert (tune_dir / "compile_default" / "metrics.json").exists()
    assert (tune_dir / "compile_default" / "stdout.log").read_text(encoding="utf-8") == ""


def test_experiment_runner_rejects_faster_quality_regression():
    tmp_path = _workspace_tmp("runner-quality")
    runner = ExperimentRunner(
        workload_command=[sys.executable, "-c", QUALITY_WRITER],
        job_name="unit-runner-quality",
        out_dir=str(tmp_path),
        max_trials=1,
        time_budget_s=5,
        thresholds=PromotionThresholds(min_speedup=0.01, require_sufficient_steps=3),
        environment={"cpu_count": 2},
        verbose=False,
    )

    report = runner.run()

    trial = report.trials[0]
    assert trial.throughput_delta > 0.5
    assert not trial.is_viable
    assert report.winner is None
    assert trial.quality_metrics["final_loss"] == 1.2
    assert any("quality regression" in reason for reason in trial.guard_failures)


def test_experiment_runner_repeats_and_rejects_gain_inside_noise_band():
    tmp_path = _workspace_tmp("runner-repeat-noise")
    runner = ExperimentRunner(
        workload_command=[sys.executable, "-c", REPEAT_NOISE_WRITER],
        job_name="unit-runner-repeat-noise",
        out_dir=str(tmp_path),
        max_trials=1,
        time_budget_s=5,
        repeat_count=2,
        thresholds=PromotionThresholds(min_speedup=0.01, require_sufficient_steps=3),
        environment={"cpu_count": 2},
        verbose=False,
    )

    report = runner.run()

    tune_dir = next(tmp_path.glob("tune-*"))
    trial = report.trials[0]
    metrics = json.loads((tune_dir / "dl_nw0_pin1" / "metrics.json").read_text(encoding="utf-8"))
    assert (tune_dir / "baseline" / "repeat_001" / "metrics.json").exists()
    assert (tune_dir / "baseline" / "repeat_002" / "metrics.json").exists()
    assert (tune_dir / "dl_nw0_pin1" / "repeat_001" / "metrics.json").exists()
    assert (tune_dir / "dl_nw0_pin1" / "repeat_002" / "metrics.json").exists()
    assert trial.repeat_count == 2
    assert trial.confidence_label == "inconclusive"
    assert not trial.is_viable
    assert report.winner is None
    assert any("noise-aware comparison" in reason for reason in trial.guard_failures)
    assert metrics["confidence_label"] == "inconclusive"


def test_experiment_runner_infers_baseline_batch_size_from_trace():
    tmp_path = _workspace_tmp("runner-batch-size")
    runner = ExperimentRunner(
        workload_command=[sys.executable, "-c", BATCH_SIZE_TRACE_WRITER],
        job_name="unit-runner-batch-size",
        out_dir=str(tmp_path),
        max_trials=1,
        safe_only=False,
        time_budget_s=5,
        warmup_steps=0,
        measure_steps=5,
        thresholds=PromotionThresholds(min_speedup=0.01, require_sufficient_steps=3),
        bottleneck_diagnosis="underutilized_gpu",
        environment={"cuda_available": True, "memory_headroom": 0.50},
        verbose=False,
    )

    report = runner.run()

    assert report.trials[0].config_id == "bs_40"
    assert report.trials[0].env_vars["FRX_BATCH_SIZE"] == "40"
    assert report.trials[0].throughput_delta > 0.3
    assert report.winner is not None


def test_experiment_runner_race_stage_only_full_benchmarks_top_candidates():
    tmp_path = _workspace_tmp("runner-race")
    runner = ExperimentRunner(
        workload_command=[sys.executable, "-c", RACE_WRITER],
        job_name="unit-runner-race",
        out_dir=str(tmp_path),
        max_trials=5,
        time_budget_s=5,
        race_promote_count=2,
        thresholds=PromotionThresholds(min_speedup=0.01, require_sufficient_steps=3),
        bottleneck_diagnosis="input_bound",
        environment={"cpu_count": 4},
        verbose=False,
    )

    report = runner.run()

    race_trials = [trial for trial in report.trials if trial.benchmark_stage == "race"]
    full_trials = [trial for trial in report.trials if trial.benchmark_stage == "full"]

    assert len(race_trials) == 5
    assert len(full_trials) == 2
    assert all(not trial.eligible_for_promotion for trial in race_trials)
    assert {trial.config_id for trial in full_trials} == {"dl_nw2_pin1", "dl_nw4_pin1"}
    assert report.winner is not None
    assert report.winner.config_id == "dl_nw4_pin1"
