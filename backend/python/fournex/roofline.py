"""Roofline model and MFU computation for Fournex.

Computes arithmetic intensity, achieved TFLOP/s, and roofline region from NCU
summary data and GPU architecture specs. Results use percentage-based estimation
from avg_dram_throughput_pct, avg_tensor_core_utilization_pct, and
avg_issue_slot_utilization_pct — so they are approximations, not exact counts.

Usage::

    from fournex.roofline import compute_roofline
    from fournex.arch_profiles import get_arch_profile

    arch = get_arch_profile("h100")
    result = compute_roofline(ncu_summary, arch)
    # result["roofline_region"] → "memory_bound" | "compute_bound"
    # result["mfu_pct"]         → 0–100 % of peak FP32 TFLOP/s
"""
from __future__ import annotations

from typing import Any


def compute_roofline(
    ncu_summary: dict[str, Any],
    arch_profile: dict[str, Any],
) -> dict[str, Any] | None:
    """Return roofline metrics derived from NCU run-summary percentages.

    Returns None when the arch profile lacks peak specs or the summary has no
    utilization data to estimate from.

    The ``estimated`` field in the output is always True because this function
    derives FLOPs and bandwidth from utilization percentages rather than raw
    instruction/byte counts. Results are directionally correct but not cycle-
    accurate.

    Region classification:
    - ``"memory_bound"``  — arithmetic intensity below the ridge point
    - ``"compute_bound"`` — arithmetic intensity at or above the ridge point

    MFU is always expressed relative to ``peak_fp32_tflops``, even for TC-
    dominated kernels, so values above 100% are possible when tensor cores are
    used (they exceed FP32 throughput).
    """
    peak_fp32: float | None = arch_profile.get("peak_fp32_tflops")
    peak_fp16: float | None = arch_profile.get("peak_fp16_tflops")
    peak_bw: float | None = arch_profile.get("peak_memory_bw_gbps")

    if not peak_fp32 or not peak_bw:
        return None

    dram_pct: float = ncu_summary.get("avg_dram_throughput_pct") or 0.0
    tc_pct: float = ncu_summary.get("avg_tensor_core_utilization_pct") or 0.0
    isu_pct: float = ncu_summary.get("avg_issue_slot_utilization_pct") or 0.0

    if dram_pct == 0.0 and tc_pct == 0.0 and isu_pct == 0.0:
        return None

    # Achieved bandwidth (GB/s)
    achieved_bw_gbps: float = dram_pct / 100.0 * peak_bw

    # Achieved compute (TFLOP/s): use TC path when tensor cores are active,
    # otherwise use FP32 FMA throughput as a proxy
    if tc_pct > 5.0 and peak_fp16:
        achieved_tflops: float = tc_pct / 100.0 * peak_fp16
        peak_tflops: float = peak_fp16
    else:
        achieved_tflops = isu_pct / 100.0 * peak_fp32
        peak_tflops = peak_fp32

    # Arithmetic intensity (FLOPs / byte)
    # = (TFLOP/s) / (GB/s) × 1000  →  (10^12 FLOP/s) / (10^9 B/s) = 1000 FLOP/B
    if achieved_bw_gbps > 0.0:
        arithmetic_intensity: float | None = round(
            achieved_tflops / achieved_bw_gbps * 1000.0, 4
        )
        # Ridge point: FLOPs/byte where bandwidth ceiling equals compute ceiling
        ridge_ai: float = peak_tflops / peak_bw * 1000.0
        region: str = "memory_bound" if arithmetic_intensity < ridge_ai else "compute_bound"
    else:
        arithmetic_intensity = None   # pure compute — no detectable DRAM traffic
        region = "compute_bound"

    # Roofline ceiling: tightest bound at this arithmetic intensity
    if arithmetic_intensity is not None:
        roofline_ceiling: float | None = round(
            min(peak_tflops, peak_bw * arithmetic_intensity / 1000.0), 4
        )
    else:
        roofline_ceiling = peak_tflops

    # MFU relative to peak FP32 (> 100% when TC path exceeds FP32 ceiling)
    mfu_pct: float = round(achieved_tflops / peak_fp32 * 100.0, 2)
    memory_utilization_pct: float = round(dram_pct, 2)

    return {
        "arithmetic_intensity": arithmetic_intensity,
        "achieved_tflops": round(achieved_tflops, 4),
        "peak_tflops": peak_tflops,
        "peak_bw_gbps": peak_bw,
        "mfu_pct": mfu_pct,
        "roofline_region": region,
        "memory_utilization_pct": memory_utilization_pct,
        "roofline_ceiling_tflops": roofline_ceiling,
        "estimated": True,
    }
