from __future__ import annotations

from typing import Any

from ..actions import CandidateConfig
from . import batch_size, dataloader, memory, mixed_precision, runtime


_BOTTLENECK_ALIASES = {
    "input_pipeline_bound": "input_bound",
    "input_pipeline": "input_bound",
    "dataloader_bound": "input_bound",
    "host_to_device_bound": "copy_bound",
    "h2d_bound": "copy_bound",
    "small_kernel_overhead": "launch_bound",
    "kernel_launch_bound": "launch_bound",
    "memory_bound": "memory_pressure",
}


def generate_all_candidates(
    environment: dict[str, Any] | None = None,
    baseline_batch_size: int | None = None,
    max_total: int = 12,
    safe_only: bool = True,
    bottleneck_diagnosis: str | dict[str, Any] | None = None,
) -> list[CandidateConfig]:
    """
    Generate candidate configs.

    With a bottleneck diagnosis, emit candidates focused on that bottleneck.
    Without one, fall back to the conservative generic order used by Phase 1.
    """
    bottleneck = _normalize_bottleneck(bottleneck_diagnosis)
    if bottleneck:
        return _generate_for_bottleneck(
            bottleneck=bottleneck,
            environment=environment,
            baseline_batch_size=baseline_batch_size,
            max_total=max_total,
            safe_only=safe_only,
        )

    candidates: list[CandidateConfig] = []
    candidates.extend(dataloader.generate_candidates(environment=environment, max_candidates=8))

    if not safe_only and len(candidates) < max_total:
        candidates.extend(
            batch_size.generate_candidates(
                baseline_batch_size=baseline_batch_size,
                environment=environment,
                max_candidates=max_total - len(candidates),
            )
        )

    if not safe_only and len(candidates) < max_total:
        candidates.extend(
            mixed_precision.generate_candidates(environment=environment)[
                : max_total - len(candidates)
            ]
        )

    return _dedupe_candidates(candidates)[:max_total]


def _generate_for_bottleneck(
    *,
    bottleneck: str,
    environment: dict[str, Any] | None,
    baseline_batch_size: int | None,
    max_total: int,
    safe_only: bool,
) -> list[CandidateConfig]:
    candidates: list[CandidateConfig] = []

    if bottleneck == "input_bound":
        candidates.extend(dataloader.generate_candidates(environment=environment, max_candidates=max_total))
    elif bottleneck == "copy_bound":
        candidates.extend(_copy_bound_candidates(environment, max_total))
    elif bottleneck == "launch_bound":
        if not safe_only:
            candidates.extend(runtime.generate_launch_candidates(environment=environment, max_candidates=max_total))
    elif bottleneck == "memory_pressure":
        candidates.extend(memory.generate_allocator_candidates(max_candidates=max_total))
        if not safe_only and len(candidates) < max_total:
            candidates.extend(
                mixed_precision.generate_candidates(environment=environment)[
                    : max_total - len(candidates)
                ]
            )
    elif bottleneck == "underutilized_gpu":
        if not safe_only:
            candidates.extend(
                batch_size.generate_candidates(
                    baseline_batch_size=baseline_batch_size,
                    environment=environment,
                    max_candidates=max_total,
                )
            )
            if len(candidates) < max_total:
                candidates.extend(
                    mixed_precision.generate_candidates(environment=environment)[
                        : max_total - len(candidates)
                    ]
                )
            if len(candidates) < max_total:
                candidates.extend(
                    runtime.generate_launch_candidates(environment=environment)[
                        : max_total - len(candidates)
                    ]
                )
    else:
        candidates.extend(dataloader.generate_candidates(environment=environment, max_candidates=max_total))

    return _dedupe_candidates(candidates)[:max_total]


def _copy_bound_candidates(
    environment: dict[str, Any] | None,
    max_total: int,
) -> list[CandidateConfig]:
    configs = dataloader.generate_candidates(
        environment=environment,
        max_candidates=max(max_total * 2, 8),
    )
    return [
        config
        for config in configs
        if config.env_vars.get("FRX_PIN_MEMORY") == "true"
    ][:max_total]


def _normalize_bottleneck(diagnosis: str | dict[str, Any] | None) -> str | None:
    if diagnosis is None:
        return None

    label: Any
    if isinstance(diagnosis, str):
        label = diagnosis
    elif "primary_bottleneck" in diagnosis:
        label = diagnosis.get("primary_bottleneck")
    elif "diagnosis" in diagnosis and isinstance(diagnosis["diagnosis"], dict):
        label = diagnosis["diagnosis"].get("primary_bottleneck")
    elif "steady_state" in diagnosis and isinstance(diagnosis["steady_state"], dict):
        label = _normalize_bottleneck(diagnosis["steady_state"])
    elif "run" in diagnosis and isinstance(diagnosis["run"], dict):
        label = _normalize_bottleneck(diagnosis["run"])
    else:
        label = None

    if not label:
        return None
    normalized = str(label).strip().lower().replace("-", "_")
    return _BOTTLENECK_ALIASES.get(normalized, normalized)


def _dedupe_candidates(candidates: list[CandidateConfig]) -> list[CandidateConfig]:
    seen: set[str] = set()
    unique: list[CandidateConfig] = []
    for candidate in candidates:
        if candidate.config_id in seen:
            continue
        seen.add(candidate.config_id)
        unique.append(candidate)
    return unique
