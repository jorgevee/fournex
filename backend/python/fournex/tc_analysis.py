"""Tensor Core efficiency analysis for Fournex.

Answers per-kernel: Are tensor cores firing? If not, why? Is there an
easy path to engage them (mixed precision, shape alignment)?

Key output fields per kernel
----------------------------
``tc_active``                  True when TC pipe is measurably busy (> 5 %)
``tc_eligible``                True when the arch has TC units AND the kernel
                               seems compute-intensive enough to benefit
``fallback_to_cuda_cores``     True when TC-eligible but CUDA FP32 FMA path
                               is carrying the load instead
``mixed_precision_active``     True when TC utilization implies FP16/BF16 ops
``mixed_precision_opportunity``True when arch supports BF16/FP16 but TC is idle
                               and mixed precision is not enabled in environment
``efficiency_label``           "efficient" | "underutilized" | "inactive" | "no_data"
``diagnoses``                  list of human-readable findings

Workload-level summary
----------------------
``summarize_tc_analysis(per_kernel_results, arch_profile, environment)`` rolls up
the per-kernel list into a single workload-level dict.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .kernel_inspector import KernelLaunchSummary

# ── Thresholds ────────────────────────────────────────────────────────────────

_TC_ACTIVE_THRESHOLD = 5.0       # tc_pct above this → TC pipe is firing
_TC_EFFICIENT_THRESHOLD = 50.0   # tc_pct above this → TC is well-used
_ISU_COMPUTE_THRESHOLD = 20.0    # isu_pct above this → kernel is compute-intensive
_MP_ACTIVE_THRESHOLD = 10.0      # tc_pct above this → mixed precision is likely active


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_tc_efficiency(
    summary: KernelLaunchSummary,
    arch_profile: dict[str, Any],
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Diagnose tensor core efficiency for a single kernel.

    Parameters
    ----------
    summary:      KernelLaunchSummary from parse_nsight_compute_csv_text()
    arch_profile: from get_arch_profile(); provides bf16_supported, tensor_core_min_dim
    environment:  run environment dict (mixed_precision, framework, etc.)
    """
    env = environment or {}
    tc_pct: float | None = summary.tensor_core_utilization_pct
    isu_pct: float = summary.issue_slot_utilization_pct or 0.0
    dram_pct: float = summary.dram_throughput_pct or 0.0

    # Does this architecture have tensor cores?
    arch_has_tc = "tensor_core_min_dim" in arch_profile
    bf16_supported: bool = bool(arch_profile.get("bf16_supported", False))
    fp8_supported: bool = bool(arch_profile.get("fp8_supported", False))

    # Is the kernel compute-intensive (likely a GEMM/attention candidate)?
    compute_intensive = isu_pct > _ISU_COMPUTE_THRESHOLD or (tc_pct is not None and tc_pct > 1.0)

    tc_active = (tc_pct is not None) and (tc_pct > _TC_ACTIVE_THRESHOLD)
    tc_eligible = arch_has_tc and compute_intensive

    # "Fallback to CUDA cores" pattern: eligible but running on FP32 FMA path
    fallback_to_cuda_cores = (
        tc_eligible
        and not tc_active
        and isu_pct > _ISU_COMPUTE_THRESHOLD
    )

    # Mixed precision proxy: TC use strongly implies FP16/BF16 (or FP8)
    mixed_precision_active = (tc_pct is not None) and (tc_pct > _MP_ACTIVE_THRESHOLD)

    # Opportunity: arch supports lower-precision TC but it's idle and env has not enabled AMP
    mp_already_on = bool(env.get("mixed_precision", False))
    mixed_precision_opportunity = (
        tc_eligible
        and not mixed_precision_active
        and (bf16_supported or fp8_supported)
        and not mp_already_on
    )

    # Efficiency label
    if tc_pct is None:
        efficiency_label = "no_data"
    elif tc_pct > _TC_EFFICIENT_THRESHOLD:
        efficiency_label = "efficient"
    elif tc_active:
        efficiency_label = "underutilized"
    elif tc_eligible:
        efficiency_label = "inactive"
    else:
        efficiency_label = "no_data"

    diagnoses = _build_diagnoses(
        tc_active=tc_active,
        tc_eligible=tc_eligible,
        tc_pct=tc_pct,
        isu_pct=isu_pct,
        dram_pct=dram_pct,
        fallback_to_cuda_cores=fallback_to_cuda_cores,
        mixed_precision_opportunity=mixed_precision_opportunity,
        bf16_supported=bf16_supported,
        fp8_supported=fp8_supported,
        arch_has_tc=arch_has_tc,
    )

    return {
        "tc_active": tc_active,
        "tc_eligible": tc_eligible,
        "tc_utilization_pct": tc_pct,
        "fallback_to_cuda_cores": fallback_to_cuda_cores,
        "mixed_precision_active": mixed_precision_active,
        "mixed_precision_opportunity": mixed_precision_opportunity,
        "efficiency_label": efficiency_label,
        "diagnoses": diagnoses,
    }


def summarize_tc_analysis(
    per_kernel: list[dict[str, Any]],
    arch_profile: dict[str, Any],
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Workload-level TC summary rolled up from per-kernel tc_analysis dicts.

    Parameters
    ----------
    per_kernel:   list of dicts returned by analyze_tc_efficiency() (one per kernel)
    arch_profile: GPU arch profile
    environment:  run environment dict
    """
    env = environment or {}
    if not per_kernel:
        return {
            "kernels_with_tc_data": 0,
            "kernels_tc_active": 0,
            "kernels_tc_eligible_inactive": 0,
            "kernels_fallback_to_cuda_cores": 0,
            "avg_tc_utilization_pct": None,
            "any_mixed_precision_opportunity": False,
            "overall_efficiency_label": "no_data",
            "top_finding": None,
        }

    with_data = [k for k in per_kernel if k["tc_utilization_pct"] is not None]
    tc_active_count = sum(1 for k in per_kernel if k["tc_active"])
    eligible_inactive = sum(1 for k in per_kernel if k["tc_eligible"] and not k["tc_active"])
    fallback_count = sum(1 for k in per_kernel if k["fallback_to_cuda_cores"])
    any_mp_opportunity = any(k["mixed_precision_opportunity"] for k in per_kernel)

    avg_tc_pct: float | None = None
    if with_data:
        avg_tc_pct = round(
            sum(k["tc_utilization_pct"] for k in with_data) / len(with_data), 2
        )

    # Overall label: worst-case across kernels weighted by how many are affected
    if not with_data:
        overall_label = "no_data"
    elif fallback_count > 0:
        overall_label = "inactive"
    elif eligible_inactive > 0:
        overall_label = "inactive"
    elif avg_tc_pct is not None and avg_tc_pct > _TC_EFFICIENT_THRESHOLD:
        overall_label = "efficient"
    elif tc_active_count > 0:
        overall_label = "underutilized"
    else:
        overall_label = "no_data"

    top_finding = _top_workload_finding(
        fallback_count=fallback_count,
        eligible_inactive=eligible_inactive,
        any_mp_opportunity=any_mp_opportunity,
        avg_tc_pct=avg_tc_pct,
        arch_profile=arch_profile,
        env=env,
    )

    return {
        "kernels_with_tc_data": len(with_data),
        "kernels_tc_active": tc_active_count,
        "kernels_tc_eligible_inactive": eligible_inactive,
        "kernels_fallback_to_cuda_cores": fallback_count,
        "avg_tc_utilization_pct": avg_tc_pct,
        "any_mixed_precision_opportunity": any_mp_opportunity,
        "overall_efficiency_label": overall_label,
        "top_finding": top_finding,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_diagnoses(
    *,
    tc_active: bool,
    tc_eligible: bool,
    tc_pct: float | None,
    isu_pct: float,
    dram_pct: float,
    fallback_to_cuda_cores: bool,
    mixed_precision_opportunity: bool,
    bf16_supported: bool,
    fp8_supported: bool,
    arch_has_tc: bool,
) -> list[str]:
    findings: list[str] = []

    if not arch_has_tc:
        return findings  # no TC on this arch; nothing to diagnose

    if tc_pct is None:
        findings.append(
            "Tensor core utilization metric not available — re-run NCU with "
            "sm__pipe_tensor_cycles_active metric to diagnose TC usage."
        )
        return findings

    if fallback_to_cuda_cores:
        findings.append(
            f"CUDA core fallback detected: ISU={isu_pct:.0f}% but TC={tc_pct:.0f}%. "
            "Kernel is running matrix ops on FP32 SIMT cores instead of tensor cores."
        )

    if mixed_precision_opportunity:
        precision_str = "BF16/FP8" if fp8_supported else ("BF16" if bf16_supported else "FP16")
        findings.append(
            f"Mixed-precision opportunity: switching to {precision_str} inputs could engage "
            "tensor cores and deliver 2–16× throughput for eligible matrix operations."
        )

    if tc_active and tc_pct <= _TC_EFFICIENT_THRESHOLD:
        findings.append(
            f"Tensor cores active at {tc_pct:.0f}% — below the efficient threshold "
            f"({_TC_EFFICIENT_THRESHOLD:.0f}%). Possible causes: small tile dimensions, "
            "unaligned matrix shapes, or mixed FP32/FP16 within the kernel."
        )

    if tc_active and tc_pct > _TC_EFFICIENT_THRESHOLD:
        findings.append(
            f"Tensor cores well-utilized at {tc_pct:.0f}%."
        )

    return findings


def _top_workload_finding(
    *,
    fallback_count: int,
    eligible_inactive: int,
    any_mp_opportunity: bool,
    avg_tc_pct: float | None,
    arch_profile: dict[str, Any],
    env: dict[str, Any],
) -> str | None:
    if fallback_count > 0:
        return (
            f"{fallback_count} kernel(s) are compute-intensive but not using tensor cores "
            "— likely running on FP32 CUDA cores instead of the TC pipe."
        )
    if eligible_inactive > 0:
        return (
            f"{eligible_inactive} tensor-core-eligible kernel(s) have TC utilization below "
            f"{_TC_ACTIVE_THRESHOLD:.0f}%. Verify FP16/BF16 inputs and aligned matrix dimensions."
        )
    if any_mp_opportunity:
        bf16 = arch_profile.get("bf16_supported", False)
        dtype = "BF16" if bf16 else "FP16"
        return (
            f"Mixed-precision not enabled — enabling {dtype} could engage tensor cores "
            "and significantly improve compute throughput."
        )
    if avg_tc_pct is not None and avg_tc_pct > _TC_EFFICIENT_THRESHOLD:
        return f"Tensor cores performing well (avg {avg_tc_pct:.0f}% utilization)."
    return None
