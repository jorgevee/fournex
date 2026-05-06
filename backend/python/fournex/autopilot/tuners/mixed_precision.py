from __future__ import annotations

from typing import Any

from ..actions import AutopilotAction, CandidateConfig, TIER_VALIDATED


def generate_candidates(
    environment: dict[str, Any] | None = None,
) -> list[CandidateConfig]:
    """
    Generate Tier-1 mixed precision candidates.

    Prefers bf16 when the GPU supports it (Ampere+). Falls back to fp16.
    The workload reads FRX_AMP_DTYPE and wraps the forward pass in
    torch.autocast(dtype=...). Correctness guards validate no NaN/loss
    explosion via clean exit + sufficient step count.
    """
    configs: list[CandidateConfig] = []
    cuda_available = _cuda_available(environment)

    if not cuda_available:
        return configs

    gpu_name = str(environment.get("gpu_name", "") if environment else "").lower()
    bf16_likely = _supports_bf16(gpu_name)

    # bf16 first (preferred: numerically safer than fp16 for training)
    if bf16_likely:
        env_bf16 = {"FRX_AMP_DTYPE": "bfloat16"}
        configs.append(CandidateConfig(
            config_id="amp_bf16",
            label="amp:bf16",
            actions=[AutopilotAction(
                action_id="amp_bfloat16",
                type="mixed_precision",
                description="Enable torch.autocast with dtype=bfloat16",
                tier=TIER_VALIDATED,
                reversible=True,
                requires_user_approval=False,
                env_vars=env_bf16,
                preconditions=["cuda_available", "bf16_supported"],
                expected_benefit="higher tensor core utilization, ~20% throughput gain",
                rollback={"disable_amp": True},
                risk="low",
            )],
            env_vars=env_bf16,
            tier=TIER_VALIDATED,
        ))

    # fp16 (higher speedup but requires grad scaler — workload must handle it)
    env_fp16 = {"FRX_AMP_DTYPE": "float16"}
    configs.append(CandidateConfig(
        config_id="amp_fp16",
        label="amp:fp16",
        actions=[AutopilotAction(
            action_id="amp_float16",
            type="mixed_precision",
            description="Enable torch.autocast with dtype=float16",
            tier=TIER_VALIDATED,
            reversible=True,
            requires_user_approval=False,
            env_vars=env_fp16,
            preconditions=["cuda_available"],
            expected_benefit="higher tensor core utilization, ~20–30% throughput gain",
            rollback={"disable_amp": True},
            risk="medium",
        )],
        env_vars=env_fp16,
        tier=TIER_VALIDATED,
    ))

    return configs


def _cuda_available(environment: dict[str, Any] | None) -> bool:
    if environment is None:
        return False
    return bool(environment.get("cuda_available"))


def _supports_bf16(gpu_name: str) -> bool:
    # Ampere (A100, A10, A30, RTX 30xx) and later support bf16 natively.
    bf16_keywords = ("a100", "a10", "a30", "a40", "a6000", "a800", "h100", "h200",
                     "rtx 30", "rtx 40", "rtx 50", "l40", "l4 ", "b100", "b200")
    return any(k in gpu_name for k in bf16_keywords)
