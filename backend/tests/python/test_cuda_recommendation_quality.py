"""CUDA recommendation quality tests.

Coverage:
  1. Multi-cause ambiguous ordering: correct primary rec outranks secondary
  2. Near-threshold boundaries: strict inequalities fire correctly
  3. False-positive clean kernels: no spurious recommendations
  4. Advice specificity: triggered_by and why fields are meaningful
  5. Optional GPU smoke test: fp16 GEMM ≥10% faster than fp32
  6. Regression tradeoff severity via diff_ncu_runs
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import pytest
import fournex as at


# ── Shared infrastructure ─────────────────────────────────────────────────────

def _summary_with(
    *,
    dram: float | None = None,
    tc: float | None = None,
    l1: float | None = None,
    l2: float | None = None,
    load_sectors: float | None = None,
    isu: float | None = None,
    occ: float | None = 70.0,
    stall: str = "unknown",
    stall_pct: float = 0.0,
) -> dict:
    """Build a synthetic ncu_summary dict for direct classifier/engine testing."""
    stall_map = {
        "memory_throttle": "memory",
        "long_scoreboard": "memory",
        "barrier": "sync",
        "wait": "sync",
        "short_scoreboard": "compute",
        "dispatch": "compute",
    }
    grp = stall_map.get(stall, "other")
    return {
        "kernel_count": 5,
        "kernels_with_ncu_data": 5,
        "avg_dram_throughput_pct": dram,
        "avg_tensor_core_utilization_pct": tc,
        "avg_l1_cache_hit_rate_pct": l1,
        "avg_l2_cache_hit_rate_pct": l2,
        "avg_global_load_sectors_per_request": load_sectors,
        "avg_issue_slot_utilization_pct": isu,
        "avg_occupancy_pct": occ,
        "dominant_warp_stall": stall,
        "dominant_warp_stall_pct": stall_pct,
        "warp_stall_breakdown": {stall: stall_pct} if stall != "unknown" else {},
        "memory_stall_fraction": 1.0 if grp == "memory" else 0.0,
        "compute_stall_fraction": 1.0 if grp == "compute" else 0.0,
    }


def _recs_for(ncu_summary: dict, environment: dict | None = None) -> dict:
    from fournex.recommendations.signals import extract_ncu_signals
    from fournex.recommendations.engine import generate_recommendations

    bottlenecks = at.classify_ncu_bottlenecks(ncu_summary)
    signals = extract_ncu_signals(ncu_summary, bottlenecks, environment or {})
    result = generate_recommendations(bottlenecks, ncu_summary, signals=signals)
    return {
        "ids": {r["id"] for r in result["recommendations"]},
        "bottlenecks": [b["label"] for b in bottlenecks],
        "recommendations": result["recommendations"],
    }


def _csv(**kwargs: float | None) -> str:
    """Build an NCU long-format CSV from shorthand keyword arguments."""
    _METRICS = {
        "dram": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "tc": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "l1": "l1tex__t_sector_hit_rate.pct",
        "l2": "lts__t_sector_hit_rate.pct",
        "isu": "sm__issue_active.avg.pct_of_peak_sustained_active",
        "mem_throttle": (
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle"
        ),
        "long_scoreboard": (
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_long_scoreboard"
        ),
        "barrier": "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_barrier",
        "wait": "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_wait",
    }
    rows = ["Kernel Name,Metric Name,Metric Unit,Metric Value"]
    for key, value in kwargs.items():
        if value is not None:
            rows.append(f"ker,{_METRICS[key]},%,{value}")
    return "\n".join(rows)


# ── Section 1: Multi-cause ordering ──────────────────────────────────────────
#
# When two bottlenecks co-exist, the recommendation for the primary (higher-
# scoring) bottleneck must rank above the recommendation for the secondary one.
# Scoring: confidence = bottleneck.score.  Memory-bound: score = dram/100.
# Tensor-core: score = 1 - tc/100.  At DRAM=88%, TC=20%: 0.88 vs 0.80.

def test_memory_primary_coalescing_outranks_amp() -> None:
    """DRAM=88% (primary, score=0.88) vs TC=20% (secondary, score=0.80).
    rec_ncu_improve_coalescing must be ranked before rec_ncu_enable_amp.
    """
    ncu = _summary_with(dram=88.0, tc=20.0, stall="memory_throttle", stall_pct=45.0)
    result = _recs_for(ncu, {"mixed_precision": False})

    ids = result["ids"]
    assert "rec_ncu_improve_coalescing" in ids
    assert "rec_ncu_enable_amp" in ids

    recs = result["recommendations"]
    coalescing_rank = next(i for i, r in enumerate(recs) if r["id"] == "rec_ncu_improve_coalescing")
    amp_rank = next(i for i, r in enumerate(recs) if r["id"] == "rec_ncu_enable_amp")

    assert coalescing_rank < amp_rank, (
        f"coalescing (rank {coalescing_rank}) should beat AMP (rank {amp_rank}) "
        f"when DRAM=88% (primary) and TC=20% (secondary)"
    )


def test_cache_and_warp_memory_not_amp_not_sync() -> None:
    """L1=25%, warp stall=memory_throttle 40%, DRAM=35%.
    tiling and coalescing should appear; AMP and reduce_syncthreads should not.
    """
    ncu = _summary_with(l1=25.0, dram=35.0, stall="memory_throttle", stall_pct=40.0)
    result = _recs_for(ncu)

    ids = result["ids"]
    assert "l1_cache_thrashing" in result["bottlenecks"]
    assert "warp_stall_memory" in result["bottlenecks"]
    assert "rec_ncu_tiling_shared_mem" in ids
    assert "rec_ncu_improve_coalescing" in ids
    assert "rec_ncu_enable_amp" not in ids
    assert "rec_ncu_reduce_syncthreads" not in ids


def test_sync_stall_recommends_syncthreads_not_coalescing() -> None:
    """Barrier stalls=35%, ISU=35%, DRAM=25%.
    reduce_syncthreads and block_size/ILP should fire; coalescing should not.
    """
    ncu = _summary_with(stall="barrier", stall_pct=35.0, dram=25.0, isu=35.0)
    result = _recs_for(ncu)

    ids = result["ids"]
    assert "warp_stall_sync" in result["bottlenecks"]
    assert "low_issue_efficiency" in result["bottlenecks"]
    assert "rec_ncu_reduce_syncthreads" in ids
    assert "rec_ncu_increase_block_size" in ids
    assert "rec_ncu_improve_coalescing" not in ids


# ── Section 2: Near-threshold boundaries ─────────────────────────────────────
#
# All classifier thresholds use strict inequalities (>, <).
# Tests probe the boundary from both sides to verify the edge does not bleed.

@pytest.mark.parametrize("dram,expect_memory_bound", [
    (69.9, False),  # 69.9 is NOT > 70.0
    (70.1, True),   # 70.1 IS > 70.0
])
def test_dram_threshold_memory_bandwidth_bound(
    dram: float, expect_memory_bound: bool
) -> None:
    """memory_bandwidth_bound requires dram > 70.0 (strict)."""
    ncu = _summary_with(dram=dram, stall="memory_throttle", stall_pct=45.0)
    bottlenecks = at.classify_ncu_bottlenecks(ncu)
    labels = [b["label"] for b in bottlenecks]

    if expect_memory_bound:
        assert "memory_bandwidth_bound" in labels, (
            f"memory_bandwidth_bound should fire at dram={dram}"
        )
    else:
        assert "memory_bandwidth_bound" not in labels, (
            f"memory_bandwidth_bound should NOT fire at dram={dram}"
        )


@pytest.mark.parametrize("tc,expect_tc_underutil", [
    (30.0, False),  # 30.0 is NOT < 30.0
    (29.9, True),   # 29.9 IS < 30.0
])
def test_tc_threshold_tensor_core_underutilized(
    tc: float, expect_tc_underutil: bool
) -> None:
    """tensor_core_underutilized requires tc < 30.0 AND occ > 40.0 (strict)."""
    ncu = _summary_with(tc=tc, occ=70.0)
    bottlenecks = at.classify_ncu_bottlenecks(ncu)
    labels = [b["label"] for b in bottlenecks]

    if expect_tc_underutil:
        assert "tensor_core_underutilized" in labels, (
            f"tensor_core_underutilized should fire at tc={tc}"
        )
    else:
        assert "tensor_core_underutilized" not in labels, (
            f"tensor_core_underutilized should NOT fire at tc={tc}"
        )


@pytest.mark.parametrize("isu,expect_block_size_rec", [
    (40.0, False),  # issue_efficiency_very_low = (40.0 < 40.0) = False → rule blocked
    (39.9, True),   # issue_efficiency_very_low = (39.9 < 40.0) = True → rule fires
])
def test_isu_threshold_low_issue_rule(isu: float, expect_block_size_rec: bool) -> None:
    """rule_ncu_low_issue_efficiency needs issue_efficiency_very_low (isu < 40.0).
    At isu=40.0 the bottleneck fires but the rule's signal guard is False.
    """
    ncu = _summary_with(isu=isu, stall="short_scoreboard", stall_pct=15.0)
    result = _recs_for(ncu)

    if expect_block_size_rec:
        assert "rec_ncu_increase_block_size" in result["ids"], (
            f"block_size rec should fire at isu={isu}"
        )
    else:
        assert "rec_ncu_increase_block_size" not in result["ids"], (
            f"block_size rec should NOT fire at isu={isu} (signal guard is False)"
        )


@pytest.mark.parametrize("l1,expect_cache_thrashing", [
    (40.0, False),  # 40.0 is NOT < 40.0
    (39.9, True),   # 39.9 IS < 40.0
])
def test_l1_threshold_cache_thrashing(l1: float, expect_cache_thrashing: bool) -> None:
    """l1_cache_thrashing requires l1 < 40.0 (strict). DRAM=35% keeps memory_bandwidth_bound quiet."""
    ncu = _summary_with(l1=l1, dram=35.0)
    result = _recs_for(ncu)

    if expect_cache_thrashing:
        assert "l1_cache_thrashing" in result["bottlenecks"]
        assert "rec_ncu_tiling_shared_mem" in result["ids"]
    else:
        assert "l1_cache_thrashing" not in result["bottlenecks"]


# ── Section 3: False-positive clean kernel variants ───────────────────────────

def test_well_optimized_kernel_gets_no_recommendations() -> None:
    """Kernel with healthy metrics in every dimension: no bottleneck should fire."""
    ncu = _summary_with(
        dram=50.0,       # < 70% — memory bandwidth fine
        tc=55.0,         # > 30% — tensor core usage fine
        l1=80.0,         # > 40% — L1 cache good
        l2=85.0,         # > 50% — L2 cache good
        isu=75.0,        # > 60% — issue efficiency fine
        occ=70.0,
        stall="not_selected",
        stall_pct=8.0,   # < 20% — no dominant stall
    )
    result = _recs_for(ncu)

    assert result["ids"] == set(), (
        f"Well-optimized kernel should produce zero recommendations; got {result['ids']}"
    )


def test_low_tc_blocked_by_low_occupancy() -> None:
    """tc=5% but occ=20%: tensor_core_underutilized classifier requires occ > 40.0.
    With occ=20%, the bottleneck should NOT fire and AMP should NOT be recommended.
    """
    ncu = _summary_with(tc=5.0, occ=20.0)
    result = _recs_for(ncu, {"mixed_precision": False})

    assert "tensor_core_underutilized" not in result["bottlenecks"], (
        "Low occupancy (20%) should suppress tensor_core_underutilized"
    )
    assert "rec_ncu_enable_amp" not in result["ids"]


def test_low_l1_without_high_dram_triggers_cache_not_memory_bound() -> None:
    """L1=30% (poor locality) but DRAM=35% (low bandwidth pressure).
    l1_cache_thrashing should fire; memory_bandwidth_bound should not.
    rec_ncu_improve_coalescing (memory-bound rec) should be absent.
    """
    ncu = _summary_with(l1=30.0, dram=35.0)
    result = _recs_for(ncu)

    assert "l1_cache_thrashing" in result["bottlenecks"]
    assert "memory_bandwidth_bound" not in result["bottlenecks"]
    assert "rec_ncu_tiling_shared_mem" in result["ids"]
    assert "rec_ncu_improve_coalescing" not in result["ids"]


# ── Section 4: Advice specificity ────────────────────────────────────────────

def test_coalescing_rec_triggered_by_memory_rule() -> None:
    """rec_ncu_improve_coalescing must name a memory-bandwidth or warp-stall-memory
    rule in its triggered_by field — not a sync or TC rule.
    """
    ncu = _summary_with(dram=85.0, stall="memory_throttle", stall_pct=38.0)
    result = _recs_for(ncu)

    recs_by_id = {r["id"]: r for r in result["recommendations"]}
    rec = recs_by_id.get("rec_ncu_improve_coalescing")
    assert rec is not None, "rec_ncu_improve_coalescing should be present"

    assert rec["triggered_by"] in {
        "rule_ncu_memory_bandwidth_bound",
        "rule_ncu_warp_stall_memory",
        "rule_ncu_memory_bandwidth_no_warp_stall",
    }, f"unexpected triggered_by: {rec['triggered_by']!r}"


def test_amp_rec_why_mentions_tensor_core_or_mixed_precision() -> None:
    """rec_ncu_enable_amp's why field must reference tensor cores or mixed precision."""
    ncu = _summary_with(tc=8.0, occ=65.0)
    result = _recs_for(ncu, {"mixed_precision": False})

    recs_by_id = {r["id"]: r for r in result["recommendations"]}
    rec = recs_by_id.get("rec_ncu_enable_amp")
    assert rec is not None, "rec_ncu_enable_amp should be present"

    why = rec["why"].lower()
    assert any(kw in why for kw in ("tensor", "mixed precision", "amp", "fp16", "bf16")), (
        f"AMP why field should mention tensor cores or mixed precision; got: {rec['why']!r}"
    )


def test_coalescing_rec_why_mentions_memory() -> None:
    """rec_ncu_improve_coalescing's why field must reference DRAM or memory bandwidth."""
    ncu = _summary_with(dram=85.0, stall="memory_throttle", stall_pct=38.0)
    result = _recs_for(ncu)

    recs_by_id = {r["id"]: r for r in result["recommendations"]}
    rec = recs_by_id.get("rec_ncu_improve_coalescing")
    assert rec is not None

    why = rec["why"].lower()
    assert any(kw in why for kw in ("dram", "memory", "bandwidth", "coalescing", "access")), (
        f"Coalescing rec why should mention memory; got: {rec['why']!r}"
    )


def test_all_recs_have_required_structural_fields() -> None:
    """Every emitted recommendation must carry the required top-level fields."""
    ncu = _summary_with(
        dram=85.0, tc=8.0, stall="memory_throttle", stall_pct=38.0, l1=28.0
    )
    result = _recs_for(ncu, {"mixed_precision": False})

    required = {"id", "title", "priority", "score", "tier", "triggered_by", "why", "actions"}
    for rec in result["recommendations"]:
        missing = required - rec.keys()
        assert not missing, f"Rec {rec.get('id')!r} missing fields: {missing}"
        assert 0.0 <= rec["score"] <= 1.0
        assert rec["priority"] in {"high", "medium", "low"}
        assert rec["tier"] in {"try_now", "next", "advanced"}


# ── Section 5: Optional GPU smoke test ────────────────────────────────────────

try:
    import torch as _torch
    _CUDA_AVAILABLE = _torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA GPU not available")
def test_fp16_gemm_at_least_10pct_faster_than_fp32() -> None:
    """Verify fp16 GEMM is ≥10% faster than fp32 — the core speedup claim
    behind rec_ncu_enable_amp.  Runs only when a CUDA GPU is present.
    """
    import torch
    import time

    N = 4096
    device = "cuda"
    a32 = torch.randn(N, N, device=device, dtype=torch.float32)
    b32 = torch.randn(N, N, device=device, dtype=torch.float32)
    a16, b16 = a32.half(), b32.half()

    warmup, iters = 5, 20
    for _ in range(warmup):
        torch.matmul(a32, b32)
        torch.matmul(a16, b16)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a32, b32)
    torch.cuda.synchronize()
    t32 = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a16, b16)
    torch.cuda.synchronize()
    t16 = time.perf_counter() - t0

    speedup = (t32 - t16) / t32
    assert speedup >= 0.10, (
        f"fp16 GEMM should be ≥10% faster than fp32 on this GPU; "
        f"fp32={t32:.3f}s  fp16={t16:.3f}s  speedup={speedup:.1%}"
    )


# ── Section 6: Regression tradeoff severity (diff_ncu_runs) ─────────────────

def test_memory_improvement_with_new_sync_yields_mixed_verdict() -> None:
    """Baseline: DRAM=78%, bad caches, heavy memory stalls.
    Optimized: DRAM drops to 62% (memory_bandwidth_bound resolves),
               caches improve, but a new sync stall appears (barrier=28%).
    Expected verdict: mixed — improvement in one dimension, regression in another.
    """
    baseline = _csv(
        dram=78.0, l1=30.0, l2=40.0,
        mem_throttle=42.0, long_scoreboard=18.0,
    )
    optimized = _csv(
        dram=62.0, l1=72.0, l2=78.0,
        barrier=28.0, wait=8.0,
    )

    result = at.diff_ncu_runs(baseline, optimized)
    assert result["verdict"]["outcome"] == "mixed", (
        f"Partial improvement + new sync bottleneck should yield 'mixed'; "
        f"got {result['verdict']['outcome']!r}"
    )


def test_unchanged_metrics_with_severe_new_sync_yields_regressed_verdict() -> None:
    """Clean baseline with no bottlenecks; optimized introduces a severe sync stall
    (barrier=45%) without improving anything.  Expected verdict: regressed.
    """
    clean = _csv(dram=45.0, l1=75.0, l2=82.0, isu=72.0)
    regressed = _csv(dram=45.0, l1=75.0, l2=82.0, isu=72.0, barrier=45.0, wait=20.0)

    result = at.diff_ncu_runs(clean, regressed)
    verdict = result["verdict"]

    assert verdict["outcome"] == "regressed", (
        f"No improvement + severe new sync should be 'regressed'; "
        f"got {verdict['outcome']!r}"
    )
    assert verdict["bottlenecks_new"] > 0


@pytest.mark.parametrize("sync_pct,expect_new_sync_bottleneck", [
    (19.0, False),  # 19.0 is NOT > 20.0
    (21.0, True),   # 21.0 IS > 20.0
])
def test_sync_stall_threshold_in_diff(
    sync_pct: float, expect_new_sync_bottleneck: bool
) -> None:
    """warp_stall_sync threshold is strictly > 20.0.
    Verify the diff correctly flags/ignores the sync bottleneck at the boundary.
    """
    clean = _csv(dram=45.0, l1=75.0, l2=82.0, isu=72.0)
    modified = _csv(dram=45.0, l1=75.0, l2=82.0, isu=72.0, barrier=sync_pct)

    result = at.diff_ncu_runs(clean, modified)
    has_new_sync = "warp_stall_sync" in result["bottleneck_diff"]["new"]

    if expect_new_sync_bottleneck:
        assert has_new_sync, (
            f"sync_pct={sync_pct} should appear as a new warp_stall_sync bottleneck"
        )
    else:
        assert not has_new_sync, (
            f"sync_pct={sync_pct} is below the > 20.0 threshold; should NOT be flagged"
        )
