"""Framework Abstraction Tax — a meta-classifier over existing telemetry signals.

Estimates how much of a workload's GPU inefficiency is attributable to
framework/runtime overhead (kernel-launch fragmentation, Python dispatch, missing
graph capture / kernel fusion) rather than to hardware limits (memory bandwidth,
compute bound) or the data pipeline (input/copy/sync stalls).

This is an *aggregation + naming* layer over signals already produced by the
telemetry analysis path — it adds no new collectors. In particular the existing
``launch_bound`` classifier (see ``analysis.classify_bottlenecks``) is a partial,
unnamed version of the same idea; this module quantifies it as a single 0-100
score with ranked, evidence-backed contributors.

V1 deliberately scopes out two things (see plan): it does NOT detect whether CUDA
graphs / torch.compile are actually enabled (only infers they *would help*, flagged
``inferred=True``), and it does NOT emit a recoverable-throughput multiplier (the
score is uncalibrated against ground-truth speedups, so a precise "+Nx" claim would
be unfounded).

Usage::

    from fournex.framework_abstraction_tax import compute_framework_abstraction_tax

    tax = compute_framework_abstraction_tax(run_summary, bottlenecks)
    # tax["score"]        → 0-100
    # tax["severity"]     → "low" | "moderate" | "high"
    # tax["contributors"] → ranked list of {name, points, inferred, evidence}
    # tax is None when there is no profiler telemetry to reason about.
"""
from __future__ import annotations

from typing import Any

from .thresholds import ClassifierThresholds, DEFAULT_THRESHOLDS

VERSION = "fat_v1"

# A kernel-count-per-step of this magnitude is treated as "fully fragmented" when
# normalizing launch density into a 0-1 signal.
_KERNEL_COUNT_SATURATION = 100.0

# Shapes are considered stable (graph-capture-friendly) below this volatility.
# Uses the same threshold as ClassifierThresholds.shape_volatility_ratio default.
_STABLE_SHAPE_THRESHOLD = 0.30


def compute_framework_abstraction_tax(
    run_summary: dict[str, Any],
    bottlenecks: list[dict[str, Any]],
    *,
    thresholds: ClassifierThresholds | None = None,
) -> dict[str, Any] | None:
    """Return the framework-abstraction-tax score and contributors, or None.

    Returns ``None`` when there is no profiler telemetry to reason about
    (``profiler_windows_exported == 0``) — e.g. the NCU-only path — so callers can
    omit the block rather than report a meaningless score.

    The score is the GPU-idle fraction *not* explained by the data pipeline,
    scaled by how fragmented the kernel launch stream is. It is a heuristic and is
    not calibrated against measured speedups.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    if run_summary.get("profiler_windows_exported", 0) <= 0:
        return None

    gpu_active = _clamp01(run_summary.get("average_gpu_utilization_pct", 0.0) / 100.0)

    # Idle attributable to the data pipeline is subtracted out so input/copy/sync-
    # bound workloads are not mislabeled as framework overhead (mirrors the
    # launch_bound guard in analysis.classify_bottlenecks).
    bottleneck_map = {b.get("label"): b for b in bottlenecks}
    input_frac = _evidence(bottleneck_map, "input_bound", "avg_dataloader_fraction")
    h2d_frac = _evidence(bottleneck_map, "copy_bound", "avg_h2d_fraction")
    sync_frac = _evidence(bottleneck_map, "sync_bound", "avg_sync_fraction")

    overhead_idle = max(0.0, (1.0 - gpu_active) - input_frac - h2d_frac - sync_frac)

    small_kernel_fraction = _clamp01(run_summary.get("small_kernel_fraction", 0.0))
    kernel_count_per_step = max(0.0, float(run_summary.get("kernel_count_per_step", 0.0)))
    median_kernel_us = max(0.0, float(run_summary.get("median_cuda_kernel_duration_us", 0.0)))
    shape_volatility = _clamp01(run_summary.get("shape_volatility_ratio", 0.0))
    shapes_stable = shape_volatility < t.shape_volatility_ratio

    kcps_norm = min(1.0, kernel_count_per_step / _KERNEL_COUNT_SATURATION)
    frag_signal = _clamp01(0.6 * small_kernel_fraction + 0.4 * kcps_norm)
    # Floor at 0.5 so genuine unexplained idle still scores even without strong
    # fragmentation evidence; full fragmentation doubles the weight to 1.0.
    fragmentation_weight = 0.5 + 0.5 * frag_signal

    score = int(round(_clamp(100.0 * overhead_idle * fragmentation_weight, 0.0, 100.0)))

    if score >= t.fat_high_threshold:
        severity = "high"
    elif score >= t.fat_moderate_threshold:
        severity = "moderate"
    else:
        severity = "low"

    evidence = {
        "gpu_active_fraction": round(gpu_active, 4),
        "overhead_idle_fraction": round(overhead_idle, 4),
        "pipeline_idle_fraction": round(input_frac + h2d_frac + sync_frac, 4),
        "small_kernel_fraction": round(small_kernel_fraction, 4),
        "kernel_count_per_step": round(kernel_count_per_step, 2),
        "median_cuda_kernel_duration_us": round(median_kernel_us, 3),
        "shape_volatility_ratio": round(shape_volatility, 4),
        "shapes_stable": shapes_stable,
    }

    # Only attribute mechanisms when the tax is non-trivial; when it is "low"
    # there is nothing meaningful to explain and listing drivers would mislead.
    contributors = (
        _build_contributors(
            overhead_idle=overhead_idle,
            frag_signal=frag_signal,
            small_kernel_fraction=small_kernel_fraction,
            kernel_count_per_step=kernel_count_per_step,
            median_kernel_us=median_kernel_us,
            shape_volatility=shape_volatility,
            shapes_stable=shapes_stable,
        )
        if score >= t.fat_moderate_threshold
        else []
    )

    return {
        "score": score,
        "severity": severity,
        "contributors": contributors,
        "evidence": evidence,
        "version": VERSION,
    }


def _build_contributors(
    *,
    overhead_idle: float,
    frag_signal: float,
    small_kernel_fraction: float,
    kernel_count_per_step: float,
    median_kernel_us: float,
    shape_volatility: float,
    shapes_stable: bool,
) -> list[dict[str, Any]]:
    """Ranked mechanisms behind the tax. ``points`` are relative weights (0-100)
    used only for ordering/display — they do not sum to the score."""
    contributors: list[dict[str, Any]] = []

    # Only attribute mechanisms when there is meaningful unexplained idle; below
    # this floor the tax is "low" and listing drivers would be misleading.
    if overhead_idle < 0.05:
        return contributors

    # Measured: fragmented launch stream (small kernels and/or high launch density).
    if frag_signal > 0.10:
        contributors.append(
            {
                "name": "Kernel launch fragmentation",
                "points": int(round(100.0 * frag_signal)),
                "inferred": False,
                "evidence": {
                    "small_kernel_fraction": round(small_kernel_fraction, 4),
                    "kernel_count_per_step": round(kernel_count_per_step, 2),
                    "median_cuda_kernel_duration_us": round(median_kernel_us, 3),
                },
            }
        )

    # Inferred opportunity: stable shapes + a busy launch stream means CUDA Graphs
    # / torch.compile capture would likely collapse per-launch overhead. We do NOT
    # detect whether they are already enabled — this is phrased as an opportunity.
    if shapes_stable and kernel_count_per_step >= 30.0:
        contributors.append(
            {
                "name": "Missing graph capture (opportunity)",
                "points": int(round(100.0 * overhead_idle * (0.5 + 0.5 * frag_signal))),
                "inferred": True,
                "evidence": {
                    "shapes_stable": True,
                    "kernel_count_per_step": round(kernel_count_per_step, 2),
                    "note": "Stable shapes with a heavy launch stream — CUDA Graphs / "
                    "torch.compile capture would likely reduce launch overhead.",
                },
            }
        )

    # Inferred opportunity: a high fraction of tiny kernels often indicates unfused
    # elementwise ops that a compiler could fuse.
    if small_kernel_fraction >= 0.40:
        contributors.append(
            {
                "name": "Unfused elementwise operations (opportunity)",
                "points": int(round(100.0 * small_kernel_fraction * 0.5)),
                "inferred": True,
                "evidence": {
                    "small_kernel_fraction": round(small_kernel_fraction, 4),
                    "note": "Many small kernels — operator fusion could reduce launch "
                    "count and memory round-trips.",
                },
            }
        )

    # Measured: dynamic shapes force per-step re-dispatch and preclude graph capture.
    if shape_volatility >= _STABLE_SHAPE_THRESHOLD:
        contributors.append(
            {
                "name": "Dynamic-shape dispatch overhead",
                "points": int(round(100.0 * shape_volatility)),
                "inferred": False,
                "evidence": {
                    "shape_volatility_ratio": round(shape_volatility, 4),
                    "note": "Shapes change across steps — re-dispatch / recompilation "
                    "overhead, and graph capture is not applicable.",
                },
            }
        )

    contributors.sort(key=lambda c: c["points"], reverse=True)
    return contributors


def _evidence(bottleneck_map: dict[Any, dict[str, Any]], label: str, key: str) -> float:
    return float(bottleneck_map.get(label, {}).get("evidence", {}).get(key, 0.0) or 0.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clamp01(value: Any) -> float:
    return _clamp(float(value or 0.0), 0.0, 1.0)
