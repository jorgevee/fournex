from __future__ import annotations

from typing import Any

from ..actions import AutopilotAction, CandidateConfig, TIER_VALIDATED


# Multipliers to try relative to the baseline batch size detected from the trace.
# Stop on OOM — the runner handles this via guard failures.
_MULTIPLIERS = [1.25, 1.5, 2.0]


def generate_candidates(
    baseline_batch_size: int | None = None,
    environment: dict[str, Any] | None = None,
    max_candidates: int = 3,
) -> list[CandidateConfig]:
    """
    Generate Tier-1 batch size candidates.

    Tries 1.25×, 1.5×, 2× the baseline batch size. The workload reads
    FRX_BATCH_SIZE from env and applies it. Memory guardrail (<90%) is
    enforced post-trial via CorrectnessGuard.
    """
    base = baseline_batch_size or _detect_batch_size(environment)
    if base is None or base <= 0:
        return []

    configs: list[CandidateConfig] = []
    for mult in _MULTIPLIERS[:max_candidates]:
        candidate_bs = int(base * mult)
        if candidate_bs <= base:
            continue
        env_vars = {"FRX_BATCH_SIZE": str(candidate_bs)}
        label = f"bs:{candidate_bs}({mult:.2g}x)"
        action = AutopilotAction(
            action_id=f"batch_size_{candidate_bs}",
            type="batch_size",
            description=f"batch_size={candidate_bs} ({mult:.2g}× baseline {base})",
            tier=TIER_VALIDATED,
            reversible=True,
            requires_user_approval=False,
            env_vars=env_vars,
            preconditions=["loss_finite", "no_oom"],
            expected_benefit="higher GPU occupancy and throughput per step",
            rollback={"restore_batch_size": base},
            risk="low",
        )
        configs.append(CandidateConfig(
            config_id=f"bs_{candidate_bs}",
            label=label,
            actions=[action],
            env_vars=env_vars,
            tier=TIER_VALIDATED,
        ))

    return configs


def _detect_batch_size(environment: dict[str, Any] | None) -> int | None:
    if environment and isinstance(environment.get("batch_size"), int):
        return environment["batch_size"]
    return None
