from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

from .arch_profiles import (
    _normalize_gpu_key,
    _select_profile_override,
    load_arch_profile_overrides,
    resolve_sm_version,
)

CLASSIFIER_VERSION = "0.2.0"


@dataclass(frozen=True)
class ClassifierThresholds:
    # --- NCU bottleneck classifier (ncu_analysis.classify_ncu_bottlenecks) ---
    ncu_dram_throughput_high_pct: float = 70.0
    ncu_memory_stall_fraction_min: float = 0.50
    ncu_dominant_warp_stall_pct: float = 20.0
    ncu_l1_hit_low_pct: float = 40.0
    ncu_l2_hit_low_pct: float = 50.0
    ncu_load_sectors_per_request_high: float = 4.0
    ncu_tc_util_low_pct: float = 30.0
    ncu_tc_occupancy_ok_pct: float = 40.0
    ncu_occupancy_low_pct: float = 40.0
    ncu_eligible_warps_low: float = 1.0
    ncu_scheduler_active_low_pct: float = 40.0
    ncu_issue_slot_low_pct: float = 60.0
    # --- Telemetry bottleneck classifier (analysis.classify_bottlenecks) ---
    input_bound_ratio: float = 0.2
    copy_bound_ratio: float = 0.15
    sync_bound_ratio: float = 0.1
    underutilized_gpu_util_pct: float = 35.0
    memory_pressure_peak_ratio: float = 0.9
    shape_volatility_ratio: float = 0.3
    launch_bound_gpu_util_pct: float = 50.0
    launch_bound_stall_ratio_max: float = 0.1
    # --- Framework abstraction tax bands ---
    fat_high_threshold: float = 45.0
    fat_moderate_threshold: float = 20.0


DEFAULT_THRESHOLDS = ClassifierThresholds()

_THRESHOLD_FIELDS = {f.name for f in ClassifierThresholds.__dataclass_fields__.values()}  # type: ignore[attr-defined]


@dataclass(frozen=True)
class ResolvedThresholds:
    values: ClassifierThresholds
    source: str
    sm_version: str | None
    thresholds_hash: str


def _compute_hash(values: ClassifierThresholds) -> str:
    payload = json.dumps(asdict(values), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def resolve_thresholds(environment: dict[str, Any] | None) -> ResolvedThresholds:
    """Resolve classifier thresholds from environment, applying per-GPU overrides if present."""
    if not environment:
        h = _compute_hash(DEFAULT_THRESHOLDS)
        return ResolvedThresholds(DEFAULT_THRESHOLDS, "defaults", None, h)

    gpu_model = environment.get("gpu_model") or environment.get("gpu_type")
    sm = resolve_sm_version(gpu_model) if gpu_model else None

    raw_overrides = environment.get("arch_profile_overrides") or {}
    if isinstance(raw_overrides, (str,)):
        raw_overrides = load_arch_profile_overrides(raw_overrides)

    override_block = _select_profile_override(gpu_model, sm, raw_overrides)
    threshold_overrides: dict[str, Any] = override_block.get("classifier_thresholds", {})

    if not threshold_overrides:
        h = _compute_hash(DEFAULT_THRESHOLDS)
        return ResolvedThresholds(DEFAULT_THRESHOLDS, "defaults", sm, h)

    unknown = set(threshold_overrides) - _THRESHOLD_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown classifier_thresholds key(s): {sorted(unknown)}. "
            f"Valid keys: {sorted(_THRESHOLD_FIELDS)}"
        )

    values = replace(DEFAULT_THRESHOLDS, **threshold_overrides)
    h = _compute_hash(values)
    return ResolvedThresholds(values, "defaults+arch_overrides", sm, h)
