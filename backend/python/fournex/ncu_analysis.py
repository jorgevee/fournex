from __future__ import annotations

from typing import Any

from .kernel_inspector import KernelLaunchSummary, parse_nsight_compute_csv, parse_nsight_compute_csv_text

_MEMORY_STALL_TYPES = frozenset({"memory_throttle", "long_scoreboard", "mio", "lg", "texture"})
_COMPUTE_STALL_TYPES = frozenset({"short_scoreboard", "dispatch", "not_selected"})
_SYNC_STALL_TYPES = frozenset({"barrier", "wait"})


def derive_ncu_run_summary(summaries: list[KernelLaunchSummary]) -> dict[str, Any]:
    kernel_count = len(summaries)
    if kernel_count == 0:
        return {
            "kernel_count": 0,
            "kernels_with_ncu_data": 0,
            "avg_dram_throughput_pct": None,
            "avg_tensor_core_utilization_pct": None,
            "avg_l1_cache_hit_rate_pct": None,
            "avg_l2_cache_hit_rate_pct": None,
            "avg_issue_slot_utilization_pct": None,
            "avg_occupancy_pct": None,
            "dominant_warp_stall": "unknown",
            "dominant_warp_stall_pct": 0.0,
            "warp_stall_breakdown": {},
            "memory_stall_fraction": 0.0,
            "compute_stall_fraction": 0.0,
        }

    def _avg(values: list[float]) -> float | None:
        present = [v for v in values if v is not None]
        return round(sum(present) / len(present), 2) if present else None

    drams = [s.dram_throughput_pct for s in summaries]
    tc_utils = [s.tensor_core_utilization_pct for s in summaries]
    l1_hits = [s.l1_cache_hit_rate_pct for s in summaries]
    l2_hits = [s.l2_cache_hit_rate_pct for s in summaries]
    isus = [s.issue_slot_utilization_pct for s in summaries]
    occs = [
        s.occupancy_estimate.get("occupancy_pct")
        for s in summaries
        if isinstance(s.occupancy_estimate, dict)
    ]

    # kernels_with_ncu_data: those with at least one performance counter
    kernels_with_ncu_data = sum(
        1 for s in summaries
        if any(v is not None for v in [s.dram_throughput_pct, s.tensor_core_utilization_pct,
                                        s.l1_cache_hit_rate_pct, s.issue_slot_utilization_pct,
                                        s.dominant_warp_stall])
    )

    # Aggregate warp stall breakdown across kernels
    all_stall_types: set[str] = set()
    for s in summaries:
        all_stall_types.update(s.warp_stall_breakdown.keys())

    combined_stalls: dict[str, float] = {}
    for stall_type in all_stall_types:
        values = [s.warp_stall_breakdown.get(stall_type, 0.0) for s in summaries if s.warp_stall_breakdown]
        if values:
            combined_stalls[stall_type] = round(sum(values) / len(values), 2)

    if combined_stalls:
        dominant_stall = max(combined_stalls, key=combined_stalls.__getitem__)
        dominant_stall_pct = combined_stalls[dominant_stall]
    else:
        dominant_stall = "unknown"
        dominant_stall_pct = 0.0

    # Fraction of issue-slot cycles stalled on memory/compute reasons (absolute, 0.0–1.0).
    # Each kernel contributes its raw stall percentage sum divided by 100 so the result
    # reflects magnitude, not merely which stall category happened to be dominant.
    memory_stall_fraction = round(
        sum(
            sum(v for k, v in s.warp_stall_breakdown.items() if k in _MEMORY_STALL_TYPES)
            for s in summaries
        ) / (100.0 * kernel_count),
        4,
    )
    compute_stall_fraction = round(
        sum(
            sum(v for k, v in s.warp_stall_breakdown.items() if k in _COMPUTE_STALL_TYPES)
            for s in summaries
        ) / (100.0 * kernel_count),
        4,
    )

    return {
        "kernel_count": kernel_count,
        "kernels_with_ncu_data": kernels_with_ncu_data,
        "avg_dram_throughput_pct": _avg(drams),
        "avg_tensor_core_utilization_pct": _avg(tc_utils),
        "avg_l1_cache_hit_rate_pct": _avg(l1_hits),
        "avg_l2_cache_hit_rate_pct": _avg(l2_hits),
        "avg_issue_slot_utilization_pct": _avg(isus),
        "avg_occupancy_pct": _avg(occs),
        "dominant_warp_stall": dominant_stall,
        "dominant_warp_stall_pct": dominant_stall_pct,
        "warp_stall_breakdown": combined_stalls,
        "memory_stall_fraction": memory_stall_fraction,
        "compute_stall_fraction": compute_stall_fraction,
    }


def classify_ncu_bottlenecks(ncu_summary: dict[str, Any]) -> list[dict[str, Any]]:
    bottlenecks: list[dict[str, Any]] = []
    kernels_with_data = ncu_summary.get("kernels_with_ncu_data", 0)

    if kernels_with_data == 0:
        bottlenecks.append({
            "label": "insufficient_ncu_data",
            "score": 1.0,
            "evidence": {"kernels_with_ncu_data": 0, "kernel_count": ncu_summary.get("kernel_count", 0)},
            "worst_steps": [],
        })
        return bottlenecks

    dram = ncu_summary.get("avg_dram_throughput_pct")
    tc = ncu_summary.get("avg_tensor_core_utilization_pct")
    l1 = ncu_summary.get("avg_l1_cache_hit_rate_pct")
    l2 = ncu_summary.get("avg_l2_cache_hit_rate_pct")
    isu = ncu_summary.get("avg_issue_slot_utilization_pct")
    occ = ncu_summary.get("avg_occupancy_pct")
    mem_frac = ncu_summary.get("memory_stall_fraction", 0.0)
    dominant_stall = ncu_summary.get("dominant_warp_stall", "unknown")
    dominant_stall_pct = ncu_summary.get("dominant_warp_stall_pct", 0.0)

    if dram is not None and dram > 70.0 and mem_frac > 0.50:
        bottlenecks.append({
            "label": "memory_bandwidth_bound",
            "score": round(min(dram / 100.0, 1.0), 4),
            "evidence": {
                "avg_dram_throughput_pct": dram,
                "memory_stall_fraction": mem_frac,
                "dominant_warp_stall": dominant_stall,
            },
            "worst_steps": [],
        })

    if dominant_stall in _MEMORY_STALL_TYPES and dominant_stall_pct > 20.0:
        bottlenecks.append({
            "label": "warp_stall_memory",
            "score": round(min(dominant_stall_pct / 100.0, 1.0), 4),
            "evidence": {
                "dominant_warp_stall": dominant_stall,
                "dominant_warp_stall_pct": dominant_stall_pct,
                "warp_stall_breakdown": ncu_summary.get("warp_stall_breakdown", {}),
            },
            "worst_steps": [],
        })

    if dominant_stall in _SYNC_STALL_TYPES and dominant_stall_pct > 20.0:
        bottlenecks.append({
            "label": "warp_stall_sync",
            "score": round(min(dominant_stall_pct / 100.0, 1.0), 4),
            "evidence": {
                "dominant_warp_stall": dominant_stall,
                "dominant_warp_stall_pct": dominant_stall_pct,
            },
            "worst_steps": [],
        })

    if l1 is not None and l1 < 40.0 or (l2 is not None and l2 < 50.0):
        hit_rate = l1 if l1 is not None else l2
        bottlenecks.append({
            "label": "cache_thrashing",
            "score": round(max(0.0, 1.0 - (hit_rate or 100.0) / 100.0), 4),
            "evidence": {
                "avg_l1_cache_hit_rate_pct": l1,
                "avg_l2_cache_hit_rate_pct": l2,
            },
            "worst_steps": [],
        })

    if tc is not None and tc < 30.0 and (occ is None or occ > 40.0):
        bottlenecks.append({
            "label": "tensor_core_underutilized",
            "score": round(max(0.0, 1.0 - tc / 100.0), 4),
            "evidence": {
                "avg_tensor_core_utilization_pct": tc,
                "avg_occupancy_pct": occ,
            },
            "worst_steps": [],
        })

    if isu is not None and isu < 60.0:
        bottlenecks.append({
            "label": "low_issue_efficiency",
            "score": round(max(0.0, 1.0 - isu / 100.0), 4),
            "evidence": {
                "avg_issue_slot_utilization_pct": isu,
                "dominant_warp_stall": dominant_stall,
            },
            "worst_steps": [],
        })

    bottlenecks.sort(key=lambda b: b["score"], reverse=True)
    return bottlenecks


def analyze_ncu_csv(
    path: str,
    *,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summaries = parse_nsight_compute_csv(path)
    return _build_ncu_result(summaries, environment)


def analyze_ncu_csv_text(
    text: str,
    *,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summaries = parse_nsight_compute_csv_text(text)
    return _build_ncu_result(summaries, environment)


def _build_ncu_result(
    summaries: list[KernelLaunchSummary],
    environment: dict[str, Any] | None,
) -> dict[str, Any]:
    from .recommendations.signals import extract_ncu_signals
    from .recommendations.engine import generate_recommendations

    ncu_summary = derive_ncu_run_summary(summaries)
    bottlenecks = classify_ncu_bottlenecks(ncu_summary)
    signals = extract_ncu_signals(ncu_summary, bottlenecks, environment or {})
    rec_result = generate_recommendations(bottlenecks, ncu_summary, signals=signals)

    primary = bottlenecks[0]["label"] if bottlenecks else None
    secondary = [b["label"] for b in bottlenecks[1:3]]

    return {
        "schema": "ncu_analysis_v1",
        "kernel_count": ncu_summary["kernel_count"],
        "kernels_with_ncu_data": ncu_summary["kernels_with_ncu_data"],
        "ncu_run_summary": ncu_summary,
        "bottlenecks": bottlenecks,
        "primary_bottleneck": primary,
        "secondary_bottlenecks": secondary,
        "recommendations": rec_result["recommendations"],
        "bundles": rec_result["bundles"],
        "kernel_summaries": [s.to_dict() for s in summaries],
    }
