from __future__ import annotations

from typing import Any

from .ncu_analysis import analyze_ncu_csv_text


# (metric_key, higher_is_better)
_METRIC_CONFIGS: list[tuple[str, bool]] = [
    ("avg_dram_throughput_pct",         False),
    ("avg_tensor_core_utilization_pct", True),
    ("avg_l1_cache_hit_rate_pct",       True),
    ("avg_l2_cache_hit_rate_pct",       True),
    ("avg_issue_slot_utilization_pct",  True),
    ("avg_occupancy_pct",               True),
    ("memory_stall_fraction",           False),
    ("compute_stall_fraction",          False),
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
    verdict         = _build_verdict(bottleneck_diff)

    return {
        "schema":           "ncu_comparison_v1",
        "label_baseline":   label_baseline,
        "label_optimized":  label_optimized,
        "baseline":         baseline,
        "optimized":        optimized,
        "bottleneck_diff":  bottleneck_diff,
        "metric_deltas":    metric_deltas,
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
    for key, higher_is_better in _METRIC_CONFIGS:
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
            "baseline":  a,
            "optimized": b,
            "delta":     delta,
            "direction": direction,
        }
    return deltas


# ── Verdict ───────────────────────────────────────────────────────────────────

def _build_verdict(bottleneck_diff: dict[str, Any]) -> dict[str, Any]:
    resolved  = bottleneck_diff["resolved"]
    new       = bottleneck_diff["new"]
    persistent = bottleneck_diff["persistent"]
    improved  = bottleneck_diff["improved"]

    if resolved and not new:
        outcome = "improved"
    elif new and not resolved:
        outcome = "regressed"
    elif resolved and new:
        outcome = "mixed"
    elif improved:
        outcome = "improved"
    else:
        outcome = "neutral"

    return {
        "outcome":                  outcome,
        "bottlenecks_resolved":     len(resolved),
        "bottlenecks_new":          len(new),
        "bottlenecks_persistent":   len(persistent),
        "bottlenecks_improved":     len(improved),
    }
