"""Occupancy limiter breakdown for Fournex.

Answers per-kernel: What is capping occupancy? How far below the theoretical
ceiling is the achieved occupancy, and which constraint is responsible?

Key output fields per kernel
----------------------------
``achieved_occupancy_pct``     Measured by NCU (or None when not available)
``theoretical_occupancy_pct``  Estimated ceiling given registers, shared mem, block size
``occupancy_gap_pct``          theoretical − achieved (positive = below ceiling)
``occupancy_efficiency_pct``   achieved / theoretical × 100 (how close to ceiling)
``primary_limiter``            "registers" | "shared_memory" | "block_size" |
                               "blocks" | "unknown_threads_per_block" | None
``all_limiters``               list of all contributing limiter strings
``blocks_per_sm_limits``       per-constraint active-blocks ceiling (from estimate_occupancy)
``registers_per_thread``       launch param extracted from NCU
``shared_memory_per_block_bytes``
``threads_per_block``
``diagnosis``                  human-readable primary finding

Workload-level summary
----------------------
``summarize_occupancy_analysis(per_kernel_results)`` produces counts and averages
across all kernels plus a top-level finding.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .kernel_inspector import KernelLaunchSummary

# ── Threshold for "occupancy is low" ─────────────────────────────────────────
_LOW_OCCUPANCY_PCT = 40.0
_EFFICIENCY_GOOD_PCT = 80.0  # achieved / theoretical ≥ 80 % is acceptable


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_occupancy(summary: KernelLaunchSummary) -> dict[str, Any]:
    """Produce an occupancy breakdown for a single kernel.

    Uses ``summary.occupancy_estimate`` (computed at parse time from launch params)
    alongside ``summary.achieved_occupancy_pct`` (measured by NCU).

    When launch metadata (registers_per_thread, threads_per_block) is missing, the
    theoretical fields are None but the achieved value (from NCU) is still reported.
    """
    estimate: dict[str, Any] = (
        summary.occupancy_estimate
        if isinstance(summary.occupancy_estimate, dict)
        else {}
    )

    theoretical_pct: float | None = estimate.get("occupancy_pct")
    achieved_pct: float | None = summary.achieved_occupancy_pct
    limiting_factors: list[str] = list(estimate.get("limiting_factors") or [])
    blocks_per_sm_limits: dict[str, Any] = dict(estimate.get("blocks_per_sm_limits") or {})

    primary_limiter: str | None = limiting_factors[0] if limiting_factors else None

    # Gap: theoretical estimate minus NCU-measured achieved
    gap_pct: float | None = None
    if theoretical_pct is not None and achieved_pct is not None:
        gap_pct = round(theoretical_pct - achieved_pct, 2)

    # Efficiency: achieved as a fraction of the theoretical ceiling
    efficiency_pct: float | None = None
    if theoretical_pct is not None and theoretical_pct > 0.0 and achieved_pct is not None:
        efficiency_pct = round(achieved_pct / theoretical_pct * 100.0, 1)

    diagnosis = _make_diagnosis(
        primary_limiter=primary_limiter,
        achieved_pct=achieved_pct,
        theoretical_pct=theoretical_pct,
        gap_pct=gap_pct,
        efficiency_pct=efficiency_pct,
        registers_per_thread=summary.registers_per_thread,
        shared_memory_bytes=summary.shared_memory_per_block_bytes,
        threads_per_block=summary.threads_per_block,
    )

    return {
        "achieved_occupancy_pct": achieved_pct,
        "theoretical_occupancy_pct": theoretical_pct,
        "occupancy_gap_pct": gap_pct,
        "occupancy_efficiency_pct": efficiency_pct,
        "primary_limiter": primary_limiter,
        "all_limiters": limiting_factors,
        "blocks_per_sm_limits": blocks_per_sm_limits,
        "registers_per_thread": summary.registers_per_thread,
        "shared_memory_per_block_bytes": summary.shared_memory_per_block_bytes,
        "threads_per_block": summary.threads_per_block,
        "diagnosis": diagnosis,
    }


def summarize_occupancy_analysis(per_kernel: list[dict[str, Any]]) -> dict[str, Any]:
    """Workload-level occupancy rollup from per-kernel analyze_occupancy() results."""
    if not per_kernel:
        return {
            "kernels_with_occupancy_data": 0,
            "kernels_low_occupancy": 0,
            "avg_achieved_occupancy_pct": None,
            "avg_occupancy_gap_pct": None,
            "avg_occupancy_efficiency_pct": None,
            "limiter_counts": {},
            "dominant_limiter": None,
            "top_finding": None,
        }

    with_achieved = [k for k in per_kernel if k["achieved_occupancy_pct"] is not None]
    with_gap = [k for k in per_kernel if k["occupancy_gap_pct"] is not None]
    with_efficiency = [k for k in per_kernel if k["occupancy_efficiency_pct"] is not None]
    low_occ = [k for k in with_achieved if k["achieved_occupancy_pct"] < _LOW_OCCUPANCY_PCT]

    avg_achieved = _avg([k["achieved_occupancy_pct"] for k in with_achieved])
    avg_gap = _avg([k["occupancy_gap_pct"] for k in with_gap])
    avg_efficiency = _avg([k["occupancy_efficiency_pct"] for k in with_efficiency])

    # Count limiters
    limiter_counts: dict[str, int] = {}
    for k in per_kernel:
        for lim in k.get("all_limiters") or []:
            limiter_counts[lim] = limiter_counts.get(lim, 0) + 1
    dominant_limiter = max(limiter_counts, key=limiter_counts.__getitem__) if limiter_counts else None

    top_finding = _top_workload_finding(
        low_occ_count=len(low_occ),
        total=len(per_kernel),
        dominant_limiter=dominant_limiter,
        avg_achieved=avg_achieved,
        avg_efficiency=avg_efficiency,
    )

    return {
        "kernels_with_occupancy_data": len(with_achieved),
        "kernels_low_occupancy": len(low_occ),
        "avg_achieved_occupancy_pct": avg_achieved,
        "avg_occupancy_gap_pct": avg_gap,
        "avg_occupancy_efficiency_pct": avg_efficiency,
        "limiter_counts": limiter_counts,
        "dominant_limiter": dominant_limiter,
        "top_finding": top_finding,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

_LIMITER_MESSAGES: dict[str, str] = {
    "registers": (
        "Register count is the primary occupancy limiter. "
        "Reduce register usage or use __launch_bounds__ to allow more blocks per SM."
    ),
    "shared_memory": (
        "Shared memory footprint is the primary occupancy limiter. "
        "Reduce tile size, reuse shared buffers, or use dynamic allocation."
    ),
    "threads": (
        "Block size is limiting parallel blocks per SM. "
        "Consider smaller blocks to fit more per SM, or verify that 100% occupancy is needed."
    ),
    "blocks": (
        "Maximum blocks-per-SM limit reached. "
        "This is usually not a problem unless occupancy is also below target."
    ),
    "unknown_threads_per_block": (
        "Thread count not available — re-export NCU with launch__block_size metric "
        "for accurate occupancy analysis."
    ),
}


def _make_diagnosis(
    *,
    primary_limiter: str | None,
    achieved_pct: float | None,
    theoretical_pct: float | None,
    gap_pct: float | None,
    efficiency_pct: float | None,
    registers_per_thread: int | None,
    shared_memory_bytes: int | None,
    threads_per_block: int | None,
) -> str | None:
    if achieved_pct is None and theoretical_pct is None:
        return None

    parts: list[str] = []

    if primary_limiter and primary_limiter in _LIMITER_MESSAGES:
        parts.append(_LIMITER_MESSAGES[primary_limiter])

    # Add quantitative detail when register/shared data present
    if primary_limiter == "registers" and registers_per_thread is not None:
        parts.append(f"Current: {registers_per_thread} registers/thread.")
    elif primary_limiter == "shared_memory" and shared_memory_bytes is not None:
        kb = shared_memory_bytes / 1024.0
        parts.append(f"Current: {kb:.1f} KB shared memory/block.")

    if efficiency_pct is not None and efficiency_pct < _EFFICIENCY_GOOD_PCT:
        parts.append(
            f"Achieved occupancy ({achieved_pct:.0f}%) is {100.0 - efficiency_pct:.0f}% "
            f"below the estimated ceiling ({theoretical_pct:.0f}%)."
        )

    if achieved_pct is not None and achieved_pct < _LOW_OCCUPANCY_PCT and not parts:
        parts.append(
            f"Low occupancy ({achieved_pct:.0f}%) — consider increasing threads per block "
            "or reducing register/shared memory pressure."
        )

    return " ".join(parts) if parts else None


def _top_workload_finding(
    *,
    low_occ_count: int,
    total: int,
    dominant_limiter: str | None,
    avg_achieved: float | None,
    avg_efficiency: float | None,
) -> str | None:
    if low_occ_count > 0:
        pct_kernels = round(low_occ_count / total * 100.0)
        msg = f"{low_occ_count}/{total} kernels ({pct_kernels}%) have occupancy below {_LOW_OCCUPANCY_PCT:.0f}%."
        if dominant_limiter and dominant_limiter in _LIMITER_MESSAGES:
            msg += f" Primary limiter: {dominant_limiter.replace('_', ' ')}."
        return msg
    if avg_achieved is not None and avg_achieved >= _LOW_OCCUPANCY_PCT:
        if avg_efficiency is not None and avg_efficiency >= _EFFICIENCY_GOOD_PCT:
            return f"Occupancy is healthy (avg {avg_achieved:.0f}%, {avg_efficiency:.0f}% of ceiling)."
        if avg_efficiency is not None:
            return (
                f"Average occupancy {avg_achieved:.0f}% is {100.0 - avg_efficiency:.0f}% "
                "below the theoretical ceiling — some register or shared-memory pressure present."
            )
    return None


def _avg(values: list[float]) -> float | None:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 2) if present else None
