from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Safety tier constants — mirror the three tiers in safe_autopilot_actions.md.
TIER_SAFE = 0       # env/dataloader knobs — no validation required
TIER_VALIDATED = 1  # batch size, AMP — run correctness guards before promoting
TIER_RISKY = 2      # distributed, custom kernels — require explicit user approval


@dataclass
class AutopilotAction:
    action_id: str
    type: str
    description: str
    tier: int
    reversible: bool
    requires_user_approval: bool
    # Injected into the subprocess environment for each trial.
    env_vars: dict[str, str]
    preconditions: list[str]
    expected_benefit: str
    rollback: dict[str, Any]
    risk: str = "low"


@dataclass
class CandidateConfig:
    config_id: str
    label: str
    actions: list[AutopilotAction]
    # Merged env_vars from all constituent actions.
    env_vars: dict[str, str]
    # Highest tier of any action in this config.
    tier: int = 0


@dataclass
class TrialConfig:
    name: str
    patch: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    risk_level: str = "low"
    label: str | None = None

    @property
    def config_id(self) -> str:
        return self.name

    @property
    def display_label(self) -> str:
        return self.label or self.name

    @classmethod
    def from_candidate(cls, candidate: CandidateConfig) -> "TrialConfig":
        patch = {
            "actions": [
                {
                    "action_id": action.action_id,
                    "type": action.type,
                    "description": action.description,
                    "risk": action.risk,
                    "tier": action.tier,
                }
                for action in candidate.actions
            ],
        }
        risk = "low"
        if any(action.risk == "high" for action in candidate.actions):
            risk = "high"
        elif any(action.risk == "medium" for action in candidate.actions):
            risk = "medium"
        return cls(
            name=candidate.config_id,
            label=candidate.label,
            patch=patch,
            env=dict(candidate.env_vars),
            risk_level=risk,
        )


@dataclass
class TrialResult:
    config_id: str
    label: str
    exit_code: int
    # Core metrics extracted from derived/summary.json
    throughput_steps_per_sec: float
    avg_gpu_utilization_pct: float
    avg_step_time_ms: float
    peak_memory_ratio: float
    dominant_stall: str
    step_count: int
    # Correctness validation
    passed_guards: bool
    guard_failures: list[str]
    # Delta vs baseline, filled in after comparison
    throughput_delta: float = 0.0
    raw_summary: dict[str, Any] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)
    quality_metrics: dict[str, Any] = field(default_factory=dict)
    artifacts_path: str = ""
    artifact_paths: dict[str, str] = field(default_factory=dict)

    @property
    def is_viable(self) -> bool:
        return self.exit_code == 0 and self.passed_guards and self.throughput_steps_per_sec > 0


# Promotion thresholds — a candidate must clear all of these to be recommended.
@dataclass
class PromotionThresholds:
    min_speedup: float = 0.08          # 8% throughput improvement
    max_memory_ratio: float = 0.90     # GPU memory < 90%
    max_step_time_regression: float = 0.10  # step time not worse by >10%
    require_clean_exit: bool = True
    require_sufficient_steps: int = 3  # at least 3 measured steps
