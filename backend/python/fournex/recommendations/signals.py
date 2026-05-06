from __future__ import annotations

from typing import Any


def extract_signals(
    run_summary: dict[str, Any],
    bottlenecks: list[dict[str, Any]],
    per_step: list[dict[str, Any]],
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive boolean and numeric signals from classifier output.

    These signals are what rules are written against — not raw metrics.
    """
    env = environment or {}
    completed = [s for s in per_step if s.get("status") == "ok" and s.get("step_wall_time_ns", 0) > 0]

    bottleneck_map: dict[str, dict[str, Any]] = {b["label"]: b for b in bottlenecks}

    gpu_util = run_summary.get("average_gpu_utilization_pct", 0.0)
    mem_peak = run_summary.get("memory_pressure_peak_ratio", 0.0)
    shape_volatility = run_summary.get("shape_volatility_ratio", 0.0)
    profiler_windows = run_summary.get("profiler_windows_exported", 0)
    util_instability = run_summary.get("utilization_instability_pct", 0.0)

    input_frac = bottleneck_map.get("input_bound", {}).get("evidence", {}).get("avg_dataloader_fraction", 0.0)
    h2d_frac = bottleneck_map.get("copy_bound", {}).get("evidence", {}).get("avg_h2d_fraction", 0.0)
    sync_frac = bottleneck_map.get("sync_bound", {}).get("evidence", {}).get("avg_sync_fraction", 0.0)

    step_times = [s["step_wall_time_ns"] for s in completed]
    step_time_cv = _coefficient_of_variation(step_times)

    return {
        # ── Utilization ──────────────────────────────────────────────────────
        "low_gpu_activity": gpu_util < 50.0,
        "very_low_gpu_activity": gpu_util < 35.0,
        "gpu_util_pct": gpu_util,
        "unstable_gpu_util": util_instability > 20.0,

        # ── Input pipeline ───────────────────────────────────────────────────
        "input_pipeline_stalled": input_frac >= 0.20,
        "input_pipeline_severe": input_frac >= 0.40,
        "input_frac": input_frac,

        # ── H2D copy ─────────────────────────────────────────────────────────
        "h2d_copy_stalled": h2d_frac >= 0.15,
        "h2d_copy_severe": h2d_frac >= 0.30,
        "h2d_frac": h2d_frac,

        # ── Synchronization ──────────────────────────────────────────────────
        "sync_heavy": sync_frac >= 0.10,
        "sync_severe": sync_frac >= 0.25,
        "sync_frac": sync_frac,

        # ── Memory ───────────────────────────────────────────────────────────
        "memory_near_capacity": mem_peak >= 0.90,
        "memory_pressure_moderate": 0.75 <= mem_peak < 0.90,
        "memory_peak_ratio": mem_peak,

        # ── Shape stability ──────────────────────────────────────────────────
        "shapes_unstable": shape_volatility >= 0.30,
        "shapes_highly_unstable": shape_volatility >= 0.60,
        "shape_volatility": shape_volatility,

        # ── Kernel / profiler ────────────────────────────────────────────────
        "has_profiler_data": profiler_windows > 0,
        "profiler_windows": profiler_windows,

        # ── Step time ────────────────────────────────────────────────────────
        "step_time_variable": step_time_cv > 0.15,
        "step_time_cv": round(step_time_cv, 4),

        # ── Environment ──────────────────────────────────────────────────────
        "framework_pytorch": env.get("framework", "").lower() == "pytorch",
        "mixed_precision_enabled": bool(env.get("mixed_precision", False)),
        "is_distributed": bool(env.get("distributed", False)),
        "num_gpus": int(env.get("num_gpus", 1)),
        "gpu_type": str(env.get("gpu_type", "unknown")),
    }


def _coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    if avg == 0.0:
        return 0.0
    variance = sum((v - avg) ** 2 for v in values) / len(values)
    return (variance ** 0.5) / avg
