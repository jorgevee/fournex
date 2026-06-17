"""Adapt SakanaAI/AI-CUDA-Engineer-Archive ``NCU_Profile`` cells into Fournex.

The dataset stores each profile as a **Python-repr dict string** (single quotes,
``True/False/None``, apostrophes inside rule descriptions) — so it must be parsed
with :func:`ast.literal_eval`, not JSON. Shape::

    {'metrics': {'<Human Name>': {'unit': str, 'avg_value': float, ...}, ...},
     'rules':   {'<RuleName>':   {'type': 'WRN'|'INF'|..., 'description': str}}}

The profile is a **fixed section set** (Speed-of-Light + Occupancy + a few
instruction stats). It deliberately lacks the warp-stall breakdown, the global
sectors-per-request coalescing metric, and the tensor-pipe metric. We do not
fabricate those — we leave the corresponding :class:`KernelLaunchSummary` fields
unset so they surface downstream as *missing evidence* (and Fournex's
confidence-downgrade logic stays honest about what the profile can prove).

One trap motivates this module's existence: the generic NCU header map in
``kernel_inspector`` aliases ``"Memory Throughput" -> dram_throughput_pct``, but
in this dataset that metric is reported in **byte/second**, not %-of-peak. Routing
it through would make every kernel look memory-bound. Here we only ever fill a
``*_pct`` field from a metric whose unit is actually ``%``.
"""
from __future__ import annotations

import ast
import math
from typing import Any

from .kernel_inspector import KernelLaunchSummary

# Sakana human-readable metric name -> (KernelLaunchSummary field, required unit).
# Only metrics whose unit matches are mapped; a mismatch (e.g. byte/second
# "Memory Throughput") is skipped rather than silently coerced.
_PCT_FIELD_MAP: dict[str, str] = {
    "Achieved Occupancy": "achieved_occupancy_pct",
    "Theoretical Occupancy": "theoretical_occupancy_pct",
    "Issue Slots Busy": "issue_slot_utilization_pct",
    "SM Busy": "sm_throughput_pct",
    "Mem Busy": "memory_busy_pct",
    # "Max Bandwidth" is NCU's Speed-of-Light %-of-peak achievable DRAM bandwidth.
    # This is the only genuine DRAM-%-of-peak figure in the section set.
    "Max Bandwidth": "dram_throughput_pct",
    "L1/TEX Hit Rate": "l1_cache_hit_rate_pct",
    "L2 Hit Rate": "l2_cache_hit_rate_pct",
}

# Block-limit resource -> Fournex occupancy limiting-factor token (matches the
# tokens derive_ncu_run_summary maps to occupancy_limited_by_* labels).
_BLOCK_LIMIT_FACTORS: dict[str, str] = {
    "Block Limit Registers": "registers",
    "Block Limit Shared Mem": "shared_memory",
    "Block Limit Warps": "threads",
    "Block Limit SM": "blocks",
}

# Raw metrics worth preserving for transparency in the evidence output even though
# Fournex has no classifier rule that consumes them from this dataset.
_RAW_KEEP: dict[str, str] = {
    "Memory Throughput": "memory_throughput_bytes_per_s",
    "Mem Pipes Busy": "mem_pipes_busy_pct",
    "Avg. Active Threads Per Warp": "active_threads_per_warp",
    "Avg. Not Predicated Off Threads Per Warp": "not_predicated_off_threads_per_warp",
    "Warp Cycles Per Issued Instruction": "warp_cycles_per_issued_inst",
    "Executed Ipc Active": "executed_ipc_active",
}

# Sections Fournex's classifier can use that this dataset never provides. Reported
# so the eval can explain *why* certain diagnoses cannot be confirmed here.
ABSENT_SECTIONS: tuple[str, ...] = (
    "warp_stall_breakdown",          # -> warp_stall_memory / warp_stall_sync
    "global_load_sectors_per_request",  # -> uncoalesced_access
    "tensor_core_utilization_pct",   # -> tensor_core_underutilized
)


def parse_ncu_profile(raw: Any) -> dict | None:
    """Parse a Sakana ``NCU_Profile`` cell. Returns None if absent/unparseable."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _avg(metric: Any) -> float | None:
    if not isinstance(metric, dict):
        return None
    val = metric.get("avg_value")
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _unit(metric: Any) -> str:
    return str(metric.get("unit", "")).strip() if isinstance(metric, dict) else ""


def _occupancy_limiting_factors(metrics: dict[str, Any]) -> list[str]:
    """Resources whose block limit is the binding (minimum) one.

    NCU reports per-resource block limits; the smallest is what caps occupancy.
    Ties (e.g. registers and warps both = 2) yield multiple factors, matching NCU.
    """
    limits: dict[str, float] = {}
    for name, token in _BLOCK_LIMIT_FACTORS.items():
        val = _avg(metrics.get(name))
        if val is not None and val > 0:
            limits[token] = val
    if not limits:
        return []
    lowest = min(limits.values())
    return sorted(token for token, val in limits.items() if val == lowest)


def adapt_ncu_profile(
    profile: Any,
    *,
    kernel_name: str = "sakana_kernel",
) -> KernelLaunchSummary | None:
    """Build a :class:`KernelLaunchSummary` from a Sakana ``NCU_Profile`` cell.

    Accepts the raw repr string or an already-parsed dict. Returns None when the
    profile is missing or has no ``metrics`` block (the caller then runs
    static-only, which is correct — there is simply no measured evidence).
    """
    parsed = parse_ncu_profile(profile)
    if parsed is None:
        return None
    metrics = parsed.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return None

    fields: dict[str, Any] = {}
    for src_name, field_name in _PCT_FIELD_MAP.items():
        metric = metrics.get(src_name)
        if _unit(metric) != "%":
            continue  # byte/second "Memory Throughput" etc. never lands in a *_pct field
        val = _avg(metric)
        if val is not None:
            fields[field_name] = round(val, 4)

    raw_metrics: dict[str, float] = {}
    for src_name, key in _RAW_KEEP.items():
        val = _avg(metrics.get(src_name))
        if val is not None:
            raw_metrics[key] = round(val, 4)

    limiting_factors = _occupancy_limiting_factors(metrics)
    occupancy_estimate: dict[str, Any] = {}
    if fields.get("achieved_occupancy_pct") is not None:
        occupancy_estimate["occupancy_pct"] = fields["achieved_occupancy_pct"]
    if limiting_factors:
        occupancy_estimate["limiting_factors"] = limiting_factors
        occupancy_estimate["block_limits"] = {
            token: _avg(metrics.get(name))
            for name, token in _BLOCK_LIMIT_FACTORS.items()
            if _avg(metrics.get(name)) is not None
        }

    return KernelLaunchSummary(
        kernel_name=kernel_name,
        occupancy_estimate=occupancy_estimate,
        metrics=raw_metrics,
        source="sakana_ncu",
        **fields,
    )
