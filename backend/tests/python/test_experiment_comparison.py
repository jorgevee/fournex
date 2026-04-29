import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from autopilot_telemetry.autopilot.actions import PromotionThresholds, TrialResult
from autopilot_telemetry.autopilot.comparison import (
    aggregate_repeats,
    annotate_noise_comparison,
)


def _result(name: str, throughput: float) -> TrialResult:
    return TrialResult(
        config_id=name,
        label=name,
        exit_code=0,
        throughput_steps_per_sec=throughput,
        avg_gpu_utilization_pct=50.0,
        avg_step_time_ms=1000.0 / throughput,
        peak_memory_ratio=0.25,
        dominant_stall="none",
        step_count=5,
        passed_guards=True,
        guard_failures=[],
    )


def test_aggregate_repeats_uses_median_and_records_noise_inputs():
    aggregate = aggregate_repeats(
        config_id="candidate",
        label="candidate",
        repeats=[_result("candidate", 10.0), _result("candidate", 12.0), _result("candidate", 11.0)],
        artifacts_path="runs/tune/candidate",
    )

    assert aggregate.repeat_count == 3
    assert aggregate.throughput_steps_per_sec == 11.0
    assert aggregate.throughput_values == [10.0, 12.0, 11.0]
    assert aggregate.throughput_stddev == 1.0


def test_noise_comparison_marks_small_noisy_gain_inconclusive():
    baseline = aggregate_repeats(
        config_id="baseline",
        label="baseline",
        repeats=[_result("baseline", 100.0), _result("baseline", 120.0)],
        artifacts_path="runs/tune/baseline",
    )
    trial = aggregate_repeats(
        config_id="candidate",
        label="candidate",
        repeats=[_result("candidate", 114.0), _result("candidate", 116.0)],
        artifacts_path="runs/tune/candidate",
    )

    annotate_noise_comparison(
        baseline,
        trial,
        PromotionThresholds(min_speedup=0.01, require_above_noise_band=True),
    )

    assert trial.throughput_delta > 0
    assert trial.confidence_label == "inconclusive"
    assert not trial.passed_guards
    assert any("noise-aware comparison" in failure for failure in trial.guard_failures)


def test_noise_comparison_labels_stable_large_gain_high_confidence():
    baseline = aggregate_repeats(
        config_id="baseline",
        label="baseline",
        repeats=[_result("baseline", 100.0), _result("baseline", 101.0), _result("baseline", 100.0)],
        artifacts_path="runs/tune/baseline",
    )
    trial = aggregate_repeats(
        config_id="candidate",
        label="candidate",
        repeats=[_result("candidate", 125.0), _result("candidate", 126.0), _result("candidate", 125.0)],
        artifacts_path="runs/tune/candidate",
    )

    annotate_noise_comparison(
        baseline,
        trial,
        PromotionThresholds(min_speedup=0.05, require_above_noise_band=True),
    )

    assert trial.passed_guards
    assert trial.confidence_label == "high"
    assert trial.noise_band < 0.01
