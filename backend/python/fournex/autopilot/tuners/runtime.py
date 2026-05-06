from __future__ import annotations

from typing import Any

from ..actions import AutopilotAction, CandidateConfig, TIER_VALIDATED


def generate_launch_candidates(
    environment: dict[str, Any] | None = None,
    max_candidates: int = 3,
) -> list[CandidateConfig]:
    configs: list[CandidateConfig] = [
        _candidate(
            config_id="compile_default",
            label="compile:default",
            description="Enable torch.compile with the default mode",
            env_vars={"FRX_TORCH_COMPILE": "1"},
            action_id="runtime_torch_compile",
        ),
        _candidate(
            config_id="compile_reduce_overhead",
            label="compile:reduce-overhead",
            description="Enable torch.compile(mode='reduce-overhead')",
            env_vars={"FRX_TORCH_COMPILE": "1", "FRX_TORCH_COMPILE_MODE": "reduce-overhead"},
            action_id="runtime_torch_compile_reduce_overhead",
        ),
    ]

    if _static_shapes_likely(environment):
        configs.append(
            _candidate(
                config_id="cuda_graphs_try_static",
                label="cuda_graphs:try_static",
                description="Try CUDA graphs when shapes appear static",
                env_vars={"FRX_CUDA_GRAPHS": "try_if_static_shapes"},
                action_id="runtime_cuda_graphs_try_static",
                risk="medium",
            )
        )

    return configs[:max_candidates]


def _candidate(
    *,
    config_id: str,
    label: str,
    description: str,
    env_vars: dict[str, str],
    action_id: str,
    risk: str = "medium",
) -> CandidateConfig:
    action = AutopilotAction(
        action_id=action_id,
        type="runtime",
        description=description,
        tier=TIER_VALIDATED,
        reversible=True,
        requires_user_approval=False,
        env_vars=env_vars,
        preconditions=["quality_checks_enabled"],
        expected_benefit="reduced Python and kernel launch overhead",
        rollback={"remove_env_vars": list(env_vars)},
        risk=risk,
    )
    return CandidateConfig(
        config_id=config_id,
        label=label,
        actions=[action],
        env_vars=env_vars,
        tier=TIER_VALIDATED,
    )


def _static_shapes_likely(environment: dict[str, Any] | None) -> bool:
    if not environment:
        return True
    if "static_shapes" in environment:
        return bool(environment["static_shapes"])
    if "shapes_dynamic" in environment:
        return not bool(environment["shapes_dynamic"])
    return True
