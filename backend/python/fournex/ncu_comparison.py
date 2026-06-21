from __future__ import annotations

from typing import Any

from .ncu_analysis import analyze_ncu_csv_text


# (metric_key, higher_is_better, display_label, unit)
# unit: "%" for percentages, "" for raw values
_METRIC_CONFIGS: list[tuple[str, bool, str, str]] = [
    ("avg_dram_throughput_pct",             False, "DRAM Throughput",           "%"),
    ("avg_tensor_core_utilization_pct",     True,  "Tensor Core Utilization",   "%"),
    ("avg_l1_cache_hit_rate_pct",           True,  "L1 Hit Rate",               "%"),
    ("avg_l2_cache_hit_rate_pct",           True,  "L2 Hit Rate",               "%"),
    ("avg_global_load_sectors_per_request", False, "Load Sectors / Request",    ""),
    ("avg_issue_slot_utilization_pct",      True,  "Issue Slot Utilization",    "%"),
    ("avg_occupancy_pct",                   True,  "Achieved Occupancy",        "%"),
    ("memory_stall_fraction",               False, "Memory Stall Fraction",     ""),
]

_SCORE_IMPROVEMENT_THRESHOLD = 0.02


def diff_ncu_runs(
    baseline_csv: str,
    optimized_csv: str,
    *,
    label_baseline: str = "baseline",
    label_optimized: str = "optimized",
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Diff two Nsight Compute CSV profiles and report what improved, what regressed, and
    the per-metric deltas.

    Returns an ``ncu_comparison_v1`` dict with:
    - ``bottleneck_diff``: resolved, new, persistent, and score-improved bottlenecks
    - ``metric_deltas``: per-metric before/after/delta/direction
    - ``verdict``: outcome + bottleneck counts
    """
    baseline  = analyze_ncu_csv_text(baseline_csv,  environment=environment)
    optimized = analyze_ncu_csv_text(optimized_csv, environment=environment)

    bottleneck_diff = _diff_bottlenecks(baseline["bottlenecks"], optimized["bottlenecks"])
    metric_deltas   = _diff_metrics(baseline["ncu_run_summary"], optimized["ncu_run_summary"])
    kernel_time     = _diff_kernel_time(baseline["ncu_run_summary"], optimized["ncu_run_summary"])
    verdict         = _build_verdict(bottleneck_diff, kernel_time)

    return {
        "schema":           "ncu_comparison_v1",
        "label_baseline":   label_baseline,
        "label_optimized":  label_optimized,
        "baseline":         baseline,
        "optimized":        optimized,
        "bottleneck_diff":  bottleneck_diff,
        "metric_deltas":    metric_deltas,
        "kernel_time":      kernel_time,
        "verdict":          verdict,
    }


# ── Bottleneck diff ───────────────────────────────────────────────────────────

def _diff_bottlenecks(
    baseline_list:  list[dict[str, Any]],
    optimized_list: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_map  = {b["label"]: b["score"] for b in baseline_list}
    optimized_map = {b["label"]: b["score"] for b in optimized_list}
    baseline_labels  = set(baseline_map)
    optimized_labels = set(optimized_map)

    resolved   = sorted(baseline_labels - optimized_labels)
    new        = sorted(optimized_labels - baseline_labels)
    persistent = sorted(baseline_labels & optimized_labels)

    # Persistent bottlenecks whose score dropped meaningfully in optimized
    improved = [
        label for label in persistent
        if optimized_map[label] < baseline_map[label] - _SCORE_IMPROVEMENT_THRESHOLD
    ]

    score_deltas = {
        label: round(optimized_map[label] - baseline_map[label], 4)
        for label in persistent
    }

    return {
        "resolved":     resolved,
        "new":          new,
        "persistent":   persistent,
        "improved":     improved,
        "score_deltas": score_deltas,
    }


# ── Metric deltas ─────────────────────────────────────────────────────────────

def _diff_metrics(
    baseline_summary:  dict[str, Any],
    optimized_summary: dict[str, Any],
) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for key, higher_is_better, label, unit in _METRIC_CONFIGS:
        a = baseline_summary.get(key)
        b = optimized_summary.get(key)
        if a is None and b is None:
            continue
        delta = round(b - a, 4) if (a is not None and b is not None) else None
        if delta is None:
            direction = None
        elif abs(delta) < 0.001:
            direction = "neutral"
        elif higher_is_better:
            direction = "improved" if delta > 0 else "regressed"
        else:
            direction = "improved" if delta < 0 else "regressed"
        deltas[key] = {
            "label":     label,
            "unit":      unit,
            "baseline":  a,
            "optimized": b,
            "delta":     delta,
            "direction": direction,
        }
    return deltas


# ── Kernel GPU time ───────────────────────────────────────────────────────────

# Measured kernel-time ratio needed to call an outcome. Below this band the change
# is within profiling noise and we report "neutral" rather than over-reading it.
_KERNEL_SPEEDUP_IMPROVED  = 1.05
_KERNEL_SPEEDUP_REGRESSED = 0.95


def _diff_kernel_time(
    baseline_summary:  dict[str, Any],
    optimized_summary: dict[str, Any],
) -> dict[str, Any]:
    """Compare summed NCU kernel GPU time. The ratio is the trustworthy basis for a
    speedup verdict (unit-invariant; both runs pay the same profiling overhead)."""
    a = baseline_summary.get("total_kernel_duration_us")
    b = optimized_summary.get("total_kernel_duration_us")
    available = a is not None and b is not None and a > 0 and b > 0
    return {
        "available":    available,
        "baseline_us":  a,
        "optimized_us": b,
        # >1 means the optimized kernel is faster (spends less GPU time).
        "speedup_x":    round(a / b, 4) if available else None,
        "source":       "ncu_gpu__time_duration (profiler-measured, serialized)",
    }


# ── Verdict ───────────────────────────────────────────────────────────────────

def _build_verdict(
    bottleneck_diff: dict[str, Any],
    kernel_time: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved  = bottleneck_diff["resolved"]
    new       = bottleneck_diff["new"]
    persistent = bottleneck_diff["persistent"]
    improved  = bottleneck_diff["improved"]

    if resolved and not new:
        bottleneck_outcome = "improved"
    elif new and not resolved:
        bottleneck_outcome = "regressed"
    elif resolved and new:
        bottleneck_outcome = "mixed"
    elif improved:
        bottleneck_outcome = "improved"
    else:
        bottleneck_outcome = "neutral"

    # Prefer measured kernel GPU time as the headline outcome when available: it is
    # the question the user is actually asking ("did it get faster?"). The bottleneck
    # diff is retained separately so a disagreement (e.g. bottleneck resolved but time
    # unchanged) is visible rather than silently overridden.
    kernel_speedup = kernel_time.get("speedup_x") if kernel_time else None
    if kernel_speedup is not None:
        if kernel_speedup >= _KERNEL_SPEEDUP_IMPROVED:
            outcome = "improved"
        elif kernel_speedup <= _KERNEL_SPEEDUP_REGRESSED:
            outcome = "regressed"
        else:
            outcome = "neutral"
        basis = "kernel_gpu_time"
    else:
        outcome = bottleneck_outcome
        basis = "bottleneck_diff"

    return {
        "outcome":                  outcome,
        "basis":                    basis,
        "kernel_speedup_x":         kernel_speedup,
        "bottleneck_outcome":       bottleneck_outcome,
        "bottlenecks_resolved":     len(resolved),
        "bottlenecks_new":          len(new),
        "bottlenecks_persistent":   len(persistent),
        "bottlenecks_improved":     len(improved),
    }
