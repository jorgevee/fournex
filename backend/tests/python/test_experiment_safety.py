import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex.autopilot.actions import (
    AutopilotAction,
    CandidateConfig,
    TIER_RISKY,
    TIER_VALIDATED,
    TrialResult,
)
from fournex.autopilot.safety import SafetyPolicy, validate_candidate
from fournex.autopilot.tuners import generate_all_candidates


def _baseline(memory_ratio=0.50):
    return TrialResult(
        config_id="baseline",
        label="baseline",
        exit_code=0,
        throughput_steps_per_sec=10.0,
        avg_gpu_utilization_pct=50.0,
        avg_step_time_ms=100.0,
        peak_memory_ratio=memory_ratio,
        dominant_stall="none",
        step_count=10,
        passed_guards=True,
        guard_failures=[],
    )


def test_rejects_batch_size_when_memory_headroom_is_low():
    candidate = generate_all_candidates(
        environment={"batch_size": 64},
        baseline_batch_size=64,
        max_total=1,
        safe_only=False,
        bottleneck_diagnosis="underutilized_gpu",
    )[0]

    result = validate_candidate(candidate, baseline=_baseline(memory_ratio=0.90))

    assert not result.passed
    assert any("memory headroom" in reason for reason in result.reasons)


def test_rejects_mixed_precision_without_quality_checks():
    candidate = generate_all_candidates(
        environment={"cuda_available": True, "gpu_name": "NVIDIA A100"},
        max_total=4,
        safe_only=False,
        bottleneck_diagnosis="memory_pressure",
    )[2]

    result = validate_candidate(
        candidate,
        environment={
            "cuda_available": True,
            "gpu_name": "NVIDIA A100",
            "require_quality_checks": False,
        },
    )

    assert not result.passed
    assert any("quality checks" in reason for reason in result.reasons)


def test_rejects_cuda_graphs_for_dynamic_shapes():
    candidate = generate_all_candidates(
        environment={"static_shapes": True},
        max_total=3,
        safe_only=False,
        bottleneck_diagnosis="launch_bound",
    )[2]

    result = validate_candidate(candidate, environment={"shapes_dynamic": True})

    assert not result.passed
    assert any("dynamic" in reason for reason in result.reasons)


def test_rejects_risky_candidate_by_default():
    action = AutopilotAction(
        action_id="risky_custom_kernel",
        type="custom_kernel",
        description="Replace a kernel implementation",
        tier=TIER_RISKY,
        reversible=False,
        requires_user_approval=True,
        env_vars={"FRX_CUSTOM_KERNEL": "1"},
        preconditions=[],
        expected_benefit="unknown",
        rollback={},
        risk="high",
    )
    candidate = CandidateConfig(
        config_id="risky_custom_kernel",
        label="risky",
        actions=[action],
        env_vars=action.env_vars,
        tier=TIER_RISKY,
    )

    result = validate_candidate(candidate, policy=SafetyPolicy())

    assert not result.passed
    assert any("risky actions" in reason for reason in result.reasons)


def test_rejects_unsupported_compile_candidate():
    action = AutopilotAction(
        action_id="compile",
        type="runtime",
        description="Enable torch.compile",
        tier=TIER_VALIDATED,
        reversible=True,
        requires_user_approval=False,
        env_vars={"FRX_TORCH_COMPILE": "1"},
        preconditions=[],
        expected_benefit="reduced launch overhead",
        rollback={},
        risk="medium",
    )
    candidate = CandidateConfig(
        config_id="compile",
        label="compile",
        actions=[action],
        env_vars=action.env_vars,
        tier=TIER_VALIDATED,
    )

    result = validate_candidate(candidate, environment={"torch_compile_supported": False})

    assert not result.passed
    assert any("compile unsupported" in reason for reason in result.reasons)
