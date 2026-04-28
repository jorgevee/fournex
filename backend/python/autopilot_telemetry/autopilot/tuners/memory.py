from __future__ import annotations

from ..actions import AutopilotAction, CandidateConfig, TIER_SAFE


def generate_allocator_candidates(max_candidates: int = 2) -> list[CandidateConfig]:
    configs = [
        _allocator_candidate(
            config_id="allocator_max_split_128",
            label="allocator:max_split=128",
            value="max_split_size_mb:128",
        ),
        _allocator_candidate(
            config_id="allocator_expandable_segments",
            label="allocator:expandable_segments",
            value="expandable_segments:True",
        ),
    ]
    return configs[:max_candidates]


def _allocator_candidate(*, config_id: str, label: str, value: str) -> CandidateConfig:
    env_vars = {"PYTORCH_CUDA_ALLOC_CONF": value}
    action = AutopilotAction(
        action_id=config_id,
        type="memory_allocator",
        description=f"Set PYTORCH_CUDA_ALLOC_CONF={value}",
        tier=TIER_SAFE,
        reversible=True,
        requires_user_approval=False,
        env_vars=env_vars,
        preconditions=[],
        expected_benefit="reduced CUDA allocator fragmentation or allocation churn",
        rollback={"remove_env_vars": ["PYTORCH_CUDA_ALLOC_CONF"]},
        risk="low",
    )
    return CandidateConfig(
        config_id=config_id,
        label=label,
        actions=[action],
        env_vars=env_vars,
        tier=TIER_SAFE,
    )
