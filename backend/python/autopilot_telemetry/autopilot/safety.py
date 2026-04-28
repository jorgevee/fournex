from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actions import CandidateConfig, TIER_RISKY, TrialResult


@dataclass
class SafetyPolicy:
    allow_risky_actions: bool = False
    allow_user_approval_actions: bool = False
    require_quality_checks_for_precision: bool = True
    min_memory_headroom_for_batch_increase: float = 0.20


@dataclass
class ValidationResult:
    passed: bool
    reasons: list[str]


def validate_candidate(
    candidate: CandidateConfig,
    *,
    baseline: TrialResult | None = None,
    environment: dict[str, Any] | None = None,
    policy: SafetyPolicy | None = None,
) -> ValidationResult:
    env = environment or {}
    safety = policy or SafetyPolicy()
    reasons: list[str] = []

    if candidate.tier >= TIER_RISKY and not safety.allow_risky_actions:
        reasons.append("candidate contains risky actions and risky actions are disabled")

    for action in candidate.actions:
        if action.requires_user_approval and not safety.allow_user_approval_actions:
            reasons.append(f"{action.action_id} requires user approval")

    if "FRX_BATCH_SIZE" in candidate.env_vars:
        memory_headroom = _memory_headroom(baseline, env)
        if memory_headroom is not None and memory_headroom < safety.min_memory_headroom_for_batch_increase:
            reasons.append(
                "batch size candidate rejected: memory headroom "
                f"{memory_headroom:.0%} is below required "
                f"{safety.min_memory_headroom_for_batch_increase:.0%}"
            )

    precision = candidate.env_vars.get("FRX_AMP_DTYPE")
    if precision:
        if not bool(env.get("cuda_available")):
            reasons.append("mixed precision candidate rejected: CUDA is not available")
        if safety.require_quality_checks_for_precision and not _quality_checks_enabled(env):
            reasons.append("mixed precision candidate rejected: quality checks are required")
        if precision == "bfloat16" and not _bf16_supported(env):
            reasons.append("bf16 candidate rejected: GPU does not advertise bf16 support")

    if candidate.env_vars.get("FRX_CUDA_GRAPHS"):
        if _dynamic_shapes(env):
            reasons.append("CUDA graphs candidate rejected: workload shapes appear dynamic")
        if not bool(env.get("cuda_available", True)):
            reasons.append("CUDA graphs candidate rejected: CUDA is not available")

    if candidate.env_vars.get("FRX_TORCH_COMPILE"):
        if env.get("torch_compile_supported") is False:
            reasons.append("torch.compile candidate rejected: environment marks compile unsupported")
        if env.get("unsupported_dynamic_behavior") is True:
            reasons.append("torch.compile candidate rejected: unsupported dynamic behavior detected")

    return ValidationResult(passed=not reasons, reasons=reasons)


def _memory_headroom(baseline: TrialResult | None, environment: dict[str, Any]) -> float | None:
    if "memory_headroom" in environment:
        try:
            return float(environment["memory_headroom"])
        except (TypeError, ValueError):
            return None
    if baseline is not None and baseline.peak_memory_ratio > 0:
        return max(0.0, 1.0 - baseline.peak_memory_ratio)
    return None


def _quality_checks_enabled(environment: dict[str, Any]) -> bool:
    if "require_quality_checks" in environment:
        return bool(environment["require_quality_checks"])
    if "quality_checks_enabled" in environment:
        return bool(environment["quality_checks_enabled"])
    return True


def _dynamic_shapes(environment: dict[str, Any]) -> bool:
    if "shapes_dynamic" in environment:
        return bool(environment["shapes_dynamic"])
    if "static_shapes" in environment:
        return not bool(environment["static_shapes"])
    return False


def _bf16_supported(environment: dict[str, Any]) -> bool:
    if "bf16_supported" in environment:
        return bool(environment["bf16_supported"])
    gpu_name = str(environment.get("gpu_name", "")).lower()
    keywords = (
        "a100",
        "a10",
        "a30",
        "a40",
        "a6000",
        "a800",
        "h100",
        "h200",
        "rtx 30",
        "rtx 40",
        "rtx 50",
        "l40",
        "l4 ",
        "b100",
        "b200",
    )
    return any(keyword in gpu_name for keyword in keywords)
