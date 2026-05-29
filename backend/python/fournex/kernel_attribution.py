"""Per-kernel attribution and optimization opportunity scoring for Fournex.

Answers: *which kernels are responsible for low MFU or memory-bound behavior,
and which ones are worth fixing first?*

For each kernel the module produces:

- ``mfu_pct``              — achieved TFLOP/s as % of peak FP32
- ``arithmetic_intensity`` — FLOPs/byte (estimated from utilization %)
- ``roofline_region``      — "memory_bound" | "compute_bound"
- ``runtime_share_pct``    — % of total GPU time (when duration data present)
- ``opportunity_score``    — 0–1 composite: how much time is being wasted
- ``opportunity``          — "high" | "medium" | "low" label

Kernels are returned sorted by ``opportunity_score`` descending, so
``top_opportunities[0]`` is always the highest-leverage target.

Usage::

    from fournex.kernel_attribution import compute_kernel_attribution
    from fournex.arch_profiles import get_arch_profile

    arch = get_arch_profile("h100")
    attribution = compute_kernel_attribution(ncu_summaries, arch)
    for k in attribution["top_opportunities"]:
        print(k["kernel_name"], k["opportunity"], k["mfu_pct"])
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .kernel_inspector import KernelLaunchSummary


def compute_kernel_attribution(
    summaries: list[KernelLaunchSummary],
    arch_profile: dict[str, Any],
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute per-kernel attribution and rank by optimization opportunity.

    Parameters
    ----------
    summaries:    list of KernelLaunchSummary from parse_nsight_compute_csv_text()
    arch_profile: GPU arch profile from get_arch_profile(); must have peak_*_tflops
                  and peak_memory_bw_gbps for roofline to be computed
    environment:  optional run environment dict (mixed_precision, framework, etc.)

    Returns a dict with keys:
    - ``kernels``               — full list, sorted by opportunity_score descending
    - ``top_opportunities``     — first 5 entries from ``kernels``
    - ``has_runtime_share``     — True when kernel_duration_us was present in NCU data
    - ``total_profiled_kernels``— len(summaries)
    """
    from .roofline import compute_roofline
    from .tc_analysis import analyze_tc_efficiency
    from .occupancy_analysis import analyze_occupancy

    if not summaries:
        return {
            "kernels": [],
            "top_opportunities": [],
            "has_runtime_share": False,
            "total_profiled_kernels": 0,
        }

    attributed: list[dict[str, Any]] = []
    for s in summaries:
        kernel_ncu = {
            "avg_dram_throughput_pct": s.dram_throughput_pct,
            "avg_tensor_core_utilization_pct": s.tensor_core_utilization_pct,
            "avg_issue_slot_utilization_pct": s.issue_slot_utilization_pct,
        }
        roofline = compute_roofline(kernel_ncu, arch_profile)

        entry: dict[str, Any] = {
            "kernel_name": s.kernel_name,
            "duration_us": s.kernel_duration_us,
            "runtime_share_pct": None,  # filled below
            "dram_throughput_pct": s.dram_throughput_pct,
            "tensor_core_utilization_pct": s.tensor_core_utilization_pct,
            "achieved_occupancy_pct": s.achieved_occupancy_pct,
            "issue_slot_utilization_pct": s.issue_slot_utilization_pct,
            "mfu_pct": roofline["mfu_pct"] if roofline else None,
            "arithmetic_intensity": roofline["arithmetic_intensity"] if roofline else None,
            "roofline_region": roofline["roofline_region"] if roofline else None,
            "roofline_ceiling_tflops": roofline["roofline_ceiling_tflops"] if roofline else None,
            "opportunity_score": 0.0,   # filled below
            "opportunity": "low",       # filled below
            "tc_analysis": analyze_tc_efficiency(s, arch_profile, environment),
            "occupancy_analysis": analyze_occupancy(s),
        }
        attributed.append(entry)

    # Runtime share from duration data
    durations = [k["duration_us"] for k in attributed if k["duration_us"] is not None]
    has_runtime_share = len(durations) > 0
    total_duration = sum(durations) if durations else 0.0

    for k in attributed:
        dur = k["duration_us"]
        if has_runtime_share and dur is not None and total_duration > 0.0:
            k["runtime_share_pct"] = round(dur / total_duration * 100.0, 2)

    # Opportunity scoring and labels
    for k in attributed:
        k["opportunity_score"] = _opportunity_score(k, has_runtime_share)
        k["opportunity"] = _opportunity_label(k["opportunity_score"], has_runtime_share)

    attributed.sort(key=lambda k: k["opportunity_score"], reverse=True)

    return {
        "kernels": attributed,
        "top_opportunities": attributed[:5],
        "has_runtime_share": has_runtime_share,
        "total_profiled_kernels": len(attributed),
    }


# ── Internal scoring ──────────────────────────────────────────────────────────

def _opportunity_score(kernel: dict[str, Any], has_runtime_share: bool) -> float:
    """Compute a 0–1 score representing how much optimization leverage this kernel has.

    Formula: ``runtime_share × severity × mfu_gap``

    - ``runtime_share`` — fraction of total GPU time (or 1.0 when unavailable)
    - ``severity``      — weight by bottleneck type: memory-bound > low-MFU > other
    - ``mfu_gap``       — 1 - mfu_pct/100 (how far below peak we are)
    """
    mfu: float | None = kernel.get("mfu_pct")
    region: str | None = kernel.get("roofline_region")
    dram_pct: float = kernel.get("dram_throughput_pct") or 0.0
    occ_pct: float = kernel.get("achieved_occupancy_pct") or 0.0
    runtime_share: float | None = kernel.get("runtime_share_pct")

    # MFU gap: how far below peak compute are we?
    if mfu is not None:
        # TC kernels can exceed FP32 peak; treat those as "no FP32-based gap"
        mfu_clamped = min(mfu, 100.0)
        mfu_gap = max(0.0, 1.0 - mfu_clamped / 100.0)
    else:
        # No roofline data: rough proxy from DRAM pct + ISU
        isu: float = kernel.get("issue_slot_utilization_pct") or 0.0
        utilization = max(dram_pct, isu) / 100.0
        mfu_gap = max(0.0, 1.0 - utilization * 0.7)

    # Severity weight based on bottleneck type
    if region == "memory_bound":
        # Memory-bound kernels have the clearest fix path (tiling/fusion)
        severity = 1.0
    elif mfu is not None and mfu < 20.0:
        # Compute-bound but highly under-utilizing the compute ceiling
        severity = 0.85
    elif occ_pct > 0.0 and occ_pct < 30.0:
        # Low occupancy limiting factor
        severity = 0.7
    elif dram_pct > 60.0:
        # High memory pressure without full roofline context
        severity = 0.65
    else:
        severity = 0.4

    # Weight by runtime fraction (or treat uniformly when data absent)
    share = (runtime_share / 100.0) if runtime_share is not None else 1.0

    return round(share * severity * mfu_gap, 4)


def _opportunity_label(score: float, has_runtime_share: bool) -> str:
    """Map opportunity_score to a human-readable tier.

    Thresholds differ depending on whether runtime_share was available, since
    scores without runtime weighting saturate near 1.0 for severe kernels.
    """
    if has_runtime_share:
        if score > 0.25:
            return "high"
        if score > 0.08:
            return "medium"
        return "low"
    else:
        # No runtime weighting: score ≈ severity × mfu_gap ∈ [0, 1]
        if score > 0.55:
            return "high"
        if score > 0.25:
            return "medium"
        return "low"
