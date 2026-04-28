from __future__ import annotations

import os
from typing import Any

from ..actions import AutopilotAction, CandidateConfig, TIER_SAFE


# Grid of dataloader knobs to sweep. Guardrails are enforced during grid
# construction: we skip prefetch_factor when num_workers=0 (PyTorch raises),
# and cap num_workers at the detected CPU core count.
_NUM_WORKERS_CANDIDATES = [0, 2, 4, 8, 12, 16]
_PIN_MEMORY_CANDIDATES = [True, False]
_PREFETCH_FACTOR_CANDIDATES = [2, 4, 8]
_PERSISTENT_WORKERS_CANDIDATES = [True, False]


def generate_candidates(
    environment: dict[str, Any] | None = None,
    max_candidates: int = 8,
) -> list[CandidateConfig]:
    """
    Generate Tier-0 dataloader configs.

    Staged strategy (section 13): sweep num_workers + pin_memory first,
    then add prefetch_factor and persistent_workers for the top combos.
    This keeps trial count manageable before the full search space explodes.
    """
    cpu_count = _cpu_count(environment)

    configs: list[CandidateConfig] = []

    for nw in _NUM_WORKERS_CANDIDATES:
        if nw > cpu_count:
            continue

        for pin in _PIN_MEMORY_CANDIDATES:
            # Baseline-like config (0 workers, no pin) is handled by the runner itself.
            if nw == 0 and not pin:
                continue

            env_vars: dict[str, str] = {
                "FRX_NUM_WORKERS": str(nw),
                "FRX_PIN_MEMORY": "true" if pin else "false",
            }

            # Add persistent_workers and prefetch_factor when num_workers > 0.
            if nw > 0:
                env_vars["FRX_PERSISTENT_WORKERS"] = "true"
                env_vars["FRX_PREFETCH_FACTOR"] = "2"

            label = f"dl:nw={nw},pin={'T' if pin else 'F'}"
            action = AutopilotAction(
                action_id=f"dataloader_nw{nw}_pin{int(pin)}",
                type="dataloader",
                description=f"num_workers={nw}, pin_memory={pin}",
                tier=TIER_SAFE,
                reversible=True,
                requires_user_approval=False,
                env_vars=env_vars,
                preconditions=[],
                expected_benefit="reduced dataloader wait / higher GPU feed rate",
                rollback={"restore_original_dataloader_config": True},
                risk="low",
            )
            configs.append(CandidateConfig(
                config_id=f"dl_nw{nw}_pin{int(pin)}",
                label=label,
                actions=[action],
                env_vars=env_vars,
                tier=TIER_SAFE,
            ))

    # Then add prefetch_factor variants for the most promising num_workers.
    best_nw = min(8, cpu_count)
    for pf in _PREFETCH_FACTOR_CANDIDATES[1:]:  # skip default=2, already covered
        if best_nw == 0:
            continue
        env_vars = {
            "FRX_NUM_WORKERS": str(best_nw),
            "FRX_PIN_MEMORY": "true",
            "FRX_PERSISTENT_WORKERS": "true",
            "FRX_PREFETCH_FACTOR": str(pf),
        }
        label = f"dl:nw={best_nw},pin=T,pf={pf}"
        action = AutopilotAction(
            action_id=f"dataloader_nw{best_nw}_pf{pf}",
            type="dataloader",
            description=f"num_workers={best_nw}, pin_memory=True, prefetch_factor={pf}",
            tier=TIER_SAFE,
            reversible=True,
            requires_user_approval=False,
            env_vars=env_vars,
            preconditions=[],
            expected_benefit="reduced dataloader prefetch stall",
            rollback={"restore_original_dataloader_config": True},
            risk="low",
        )
        configs.append(CandidateConfig(
            config_id=f"dl_nw{best_nw}_pf{pf}",
            label=label,
            actions=[action],
            env_vars=env_vars,
            tier=TIER_SAFE,
        ))

    return configs[:max_candidates]


def _cpu_count(environment: dict[str, Any] | None) -> int:
    if environment and isinstance(environment.get("cpu_count"), int):
        return environment["cpu_count"]
    return os.cpu_count() or 4
