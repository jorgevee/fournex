from __future__ import annotations

from typing import Any

from .kernel_inspector import KernelLaunchSummary, parse_nsight_compute_csv, parse_nsight_compute_csv_text
from .ncu_presets import NCU_METRIC_PRESETS

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
            "avg_eligible_warps_per_scheduler": None,
            "avg_scheduler_active_pct": None,
            "avg_registers_per_thread": None,
            "avg_shared_memory_per_block_bytes": None,
            "avg_threads_per_block": None,
            "occupancy_limiting_factor_counts": {},
            "occupancy_limit_causes": [],
            "dominant_warp_stall": "unknown",
            "dominant_warp_stall_pct": 0.0,
            "warp_stall_breakdown": {},
            "kernels_with_warp_stall_data": 0,
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
    eligible_warps = [s.eligible_warps_per_scheduler for s in summaries]
    scheduler_active = [s.scheduler_active_pct for s in summaries]
    regs = [s.registers_per_thread for s in summaries]
    shared = [s.shared_memory_per_block_bytes for s in summaries]
    threads = [s.threads_per_block for s in summaries]
    occs = [
        s.achieved_occupancy_pct
        if s.achieved_occupancy_pct is not None
        else s.occupancy_estimate.get("occupancy_pct")
        for s in summaries
        if s.achieved_occupancy_pct is not None or isinstance(s.occupancy_estimate, dict)
    ]
    limiting_factor_counts: dict[str, int] = {}
    for s in summaries:
        estimate = s.occupancy_estimate if isinstance(s.occupancy_estimate, dict) else {}
        for factor in estimate.get("limiting_factors", []) or []:
            limiting_factor_counts[factor] = limiting_factor_counts.get(factor, 0) + 1
    occupancy_limit_causes = sorted(
        factor for factor, count in limiting_factor_counts.items()
        if factor in {"registers", "shared_memory", "threads", "blocks", "unknown_threads_per_block"} and count > 0
    )

    # kernels_with_ncu_data: those with at least one performance counter
    kernels_with_ncu_data = sum(
        1 for s in summaries
        if any(v is not None for v in [
            s.dram_throughput_pct,
            s.tensor_core_utilization_pct,
            s.l1_cache_hit_rate_pct,
            s.l2_cache_hit_rate_pct,
            s.issue_slot_utilization_pct,
            s.achieved_occupancy_pct,
            s.eligible_warps_per_scheduler,
            s.scheduler_active_pct,
            s.dominant_warp_stall,
        ])
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
    stall_samples = [s for s in summaries if s.warp_stall_breakdown]
    stall_sample_count = len(stall_samples)
    if stall_sample_count:
        memory_stall_fraction = round(
            sum(
                sum(v for k, v in s.warp_stall_breakdown.items() if k in _MEMORY_STALL_TYPES)
                for s in stall_samples
            ) / (100.0 * stall_sample_count),
            4,
        )
        compute_stall_fraction = round(
            sum(
                sum(v for k, v in s.warp_stall_breakdown.items() if k in _COMPUTE_STALL_TYPES)
                for s in stall_samples
            ) / (100.0 * stall_sample_count),
            4,
        )
    else:
        memory_stall_fraction = 0.0
        compute_stall_fraction = 0.0

    return {
        "kernel_count": kernel_count,
        "kernels_with_ncu_data": kernels_with_ncu_data,
        "avg_dram_throughput_pct": _avg(drams),
        "avg_tensor_core_utilization_pct": _avg(tc_utils),
        "avg_l1_cache_hit_rate_pct": _avg(l1_hits),
        "avg_l2_cache_hit_rate_pct": _avg(l2_hits),
        "avg_issue_slot_utilization_pct": _avg(isus),
        "avg_occupancy_pct": _avg(occs),
        "avg_eligible_warps_per_scheduler": _avg(eligible_warps),
        "avg_scheduler_active_pct": _avg(scheduler_active),
        "avg_registers_per_thread": _avg(regs),
        "avg_shared_memory_per_block_bytes": _avg(shared),
        "avg_threads_per_block": _avg(threads),
        "occupancy_limiting_factor_counts": limiting_factor_counts,
        "occupancy_limit_causes": occupancy_limit_causes,
        "dominant_warp_stall": dominant_stall,
        "dominant_warp_stall_pct": dominant_stall_pct,
        "warp_stall_breakdown": combined_stalls,
        "kernels_with_warp_stall_data": stall_sample_count,
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
    eligible = ncu_summary.get("avg_eligible_warps_per_scheduler")
    scheduler_active = ncu_summary.get("avg_scheduler_active_pct")
    causes = set(ncu_summary.get("occupancy_limit_causes") or [])
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

    if occ is not None and occ < 40.0:
        bottlenecks.append({
            "label": "occupancy_limited",
            "score": round(max(0.0, 1.0 - occ / 40.0), 4),
            "evidence": {
                "avg_occupancy_pct": occ,
                "occupancy_limit_causes": sorted(causes),
                "avg_registers_per_thread": ncu_summary.get("avg_registers_per_thread"),
                "avg_shared_memory_per_block_bytes": ncu_summary.get("avg_shared_memory_per_block_bytes"),
                "avg_threads_per_block": ncu_summary.get("avg_threads_per_block"),
            },
            "worst_steps": [],
        })

        cause_labels = {
            "registers": "occupancy_limited_by_registers",
            "shared_memory": "occupancy_limited_by_shared_memory",
            "threads": "occupancy_limited_by_block_size",
            "blocks": "occupancy_limited_by_block_size",
            "unknown_threads_per_block": "occupancy_limited_by_block_size",
        }
        for cause, label in cause_labels.items():
            if cause in causes:
                bottlenecks.append({
                    "label": label,
                    "score": round(max(0.0, 0.9 - occ / 80.0), 4),
                    "evidence": {
                        "avg_occupancy_pct": occ,
                        "limiting_factor": cause,
                    },
                    "worst_steps": [],
                })

    if (
        (eligible is not None and eligible < 1.0)
        or (scheduler_active is not None and scheduler_active < 40.0)
    ):
        bottlenecks.append({
            "label": "low_warp_scheduler_utilization",
            "score": round(max(
                1.0 - (eligible or 1.0) / 1.0 if eligible is not None else 0.0,
                1.0 - (scheduler_active or 40.0) / 40.0 if scheduler_active is not None else 0.0,
            ), 4),
            "evidence": {
                "avg_eligible_warps_per_scheduler": eligible,
                "avg_scheduler_active_pct": scheduler_active,
                "avg_issue_slot_utilization_pct": isu,
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
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    summaries = parse_nsight_compute_csv_text(text)
    return _build_ncu_result(summaries, environment, source_text=text)


def analyze_ncu_csv_text(
    text: str,
    *,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summaries = parse_nsight_compute_csv_text(text)
    return _build_ncu_result(summaries, environment, source_text=text)


def _build_ncu_result(
    summaries: list[KernelLaunchSummary],
    environment: dict[str, Any] | None,
    source_text: str | None = None,
) -> dict[str, Any]:
    from .recommendations.signals import extract_ncu_signals
    from .recommendations.engine import generate_recommendations

    ncu_summary = derive_ncu_run_summary(summaries)
    validation = validate_ncu_csv_text(source_text or "", summaries=summaries)
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
        "validation": validation,
        "diagnostic_scope": {
            "type": "measured_ncu",
            "confidence": "high" if ncu_summary["kernels_with_ncu_data"] else "low",
            "message": "Nsight Compute metrics are measured runtime evidence for the profiled kernels.",
        },
        "bottlenecks": bottlenecks,
        "primary_bottleneck": primary,
        "secondary_bottlenecks": secondary,
        "recommendations": rec_result["recommendations"],
        "bundles": rec_result["bundles"],
        "kernel_summaries": [s.to_dict() for s in summaries],
    }


def validate_ncu_csv_text(
    text: str,
    *,
    summaries: list[KernelLaunchSummary] | None = None,
    preset: str | None = None,
) -> dict[str, Any]:
    """Return user-facing validation for an Nsight Compute CSV export.

    The parser is intentionally tolerant, so validation is where we explain what
    was missing instead of treating every partial export as a hard failure.
    """
    clean_lines = [
        line
        for line in text.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and not line.lstrip().startswith("==")
    ]
    warnings: list[str] = []
    errors: list[str] = []

    if not clean_lines:
        errors.append("CSV is empty after removing comments and Nsight Compute status lines.")
        return _validation_payload(False, [], _missing_ncu_metrics_by_preset(set()), warnings, errors, preset)

    header = [column.strip() for column in clean_lines[0].split(",")]
    lowered_header = {column.lower() for column in header}
    if not any(column in lowered_header for column in ("kernel name", "kernel", "kernel name demangled", "name")):
        errors.append("CSV is missing a kernel name column.")
    if not any(column in lowered_header for column in ("metric name", "metric", "name")):
        errors.append("CSV is missing a metric name column.")
    if not any(column in lowered_header for column in ("metric value", "value", "avg", "average")):
        errors.append("CSV is missing a metric value column.")

    resolved_summaries = summaries if summaries is not None else parse_nsight_compute_csv_text(text)
    present = sorted({
        metric_name
        for summary in resolved_summaries
        for metric_name in summary.metrics
    })

    if not resolved_summaries:
        errors.append("No kernel rows could be parsed from the CSV.")
    elif not present:
        errors.append("Kernel rows were found, but no numeric NCU metrics could be parsed.")

    missing_by_preset = _missing_ncu_metrics_by_preset(set(present))
    if preset:
        preset_key = preset.strip().lower()
        if preset_key not in NCU_METRIC_PRESETS:
            errors.append(f"Unknown NCU preset: {preset}")
        else:
            missing = missing_by_preset[preset_key]
            if missing:
                warnings.append(f"Missing metrics for {preset_key} preset: {', '.join(missing)}")

    if missing_by_preset["memory"]:
        warnings.append("Memory diagnosis may be incomplete; memory preset metrics are missing.")
    if missing_by_preset["stalls"]:
        warnings.append("Warp stall diagnosis may be incomplete; stall preset metrics are missing.")
    if missing_by_preset["tensor"]:
        warnings.append("Tensor core diagnosis may be incomplete; tensor preset metrics are missing.")
    if missing_by_preset["occupancy"]:
        warnings.append("Occupancy diagnosis may be incomplete; occupancy preset metrics are missing.")

    return _validation_payload(not errors, present, missing_by_preset, warnings, errors, preset)


def _missing_ncu_metrics_by_preset(present: set[str]) -> dict[str, list[str]]:
    return {
        name: [
            metric
            for metric in preset.required_canonical_metrics
            if metric not in present
        ]
        for name, preset in NCU_METRIC_PRESETS.items()
    }


def _validation_payload(
    valid: bool,
    present_metrics: list[str],
    missing_by_preset: dict[str, list[str]],
    warnings: list[str],
    errors: list[str],
    preset: str | None,
) -> dict[str, Any]:
    return {
        "valid": valid,
        "preset": preset,
        "present_metrics": present_metrics,
        "missing_metrics_by_preset": missing_by_preset,
        "warnings": warnings,
        "errors": errors,
    }
