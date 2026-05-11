"""Golden recommendation tests for NCU bottleneck rules."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as at
from fournex.ncu_analysis import _build_ncu_result
from fournex.kernel_inspector import KernelLaunchSummary


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _summary_with(
    *,
    dram: float | None = None,
    tc: float | None = None,
    l1: float | None = None,
    l2: float | None = None,
    isu: float | None = None,
    occ: float | None = 70.0,
    stall: str = "unknown",
    stall_pct: float = 0.0,
    mem_frac: float = 0.0,
) -> dict:
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


# ── Memory bandwidth bound ────────────────────────────────────────────────────

def test_memory_bandwidth_bound_recommends_coalescing_and_tiling() -> None:
    ncu = _summary_with(dram=85.0, stall="memory_throttle", stall_pct=38.0, mem_frac=0.8, l1=30.0, l2=40.0)
    result = _recs_for(ncu)

    assert "memory_bandwidth_bound" in result["bottlenecks"]
    assert "rec_ncu_improve_coalescing" in result["ids"]
    assert "rec_ncu_tiling_shared_mem" in result["ids"]


def test_memory_bandwidth_bound_excludes_amp_when_dram_not_the_cause() -> None:
    # compute-dominant case — memory bandwidth is fine
    ncu = _summary_with(dram=30.0, tc=8.0, stall="not_selected", stall_pct=5.0, occ=65.0)
    result = _recs_for(ncu, {"mixed_precision": False})

    assert "tensor_core_underutilized" in result["bottlenecks"]
    # AMP should be recommended for tc underutil, NOT for memory bandwidth
    assert "rec_ncu_enable_amp" in result["ids"]
    # Coalescing should NOT fire (no memory bandwidth bottleneck)
    assert "rec_ncu_improve_coalescing" not in result["ids"]


# ── Tensor core underutilization ──────────────────────────────────────────────

def test_tensor_core_inactive_without_amp_recommends_amp_first() -> None:
    ncu = _summary_with(tc=5.0, occ=65.0)
    result = _recs_for(ncu, {"mixed_precision": False})

    assert "tensor_core_underutilized" in result["bottlenecks"]
    ids = result["ids"]
    assert "rec_ncu_enable_amp" in ids
    assert "rec_ncu_use_tensor_ops" in ids


def test_tensor_core_inactive_with_amp_skips_amp_rec() -> None:
    ncu = _summary_with(tc=12.0, occ=65.0)
    result = _recs_for(ncu, {"mixed_precision": True})

    ids = result["ids"]
    # AMP already enabled — should not recommend enabling it again
    assert "rec_ncu_enable_amp" not in ids
    # But alignment and tensor ops should still fire
    assert "rec_ncu_use_tensor_ops" in ids or "rec_ncu_align_matmul_dims" in ids


def test_tensor_core_underutilized_excludes_sync_recs() -> None:
    ncu = _summary_with(tc=8.0, occ=65.0, stall="not_selected", stall_pct=10.0)
    result = _recs_for(ncu, {"mixed_precision": False})

    ids = result["ids"]
    assert "rec_ncu_reduce_syncthreads" not in ids


# ── Warp stall sync ───────────────────────────────────────────────────────────

def test_warp_stall_sync_recommends_reduce_syncthreads() -> None:
    ncu = _summary_with(stall="barrier", stall_pct=30.0, isu=70.0)
    result = _recs_for(ncu)

    assert "warp_stall_sync" in result["bottlenecks"]
    ids = result["ids"]
    assert "rec_ncu_reduce_syncthreads" in ids
    assert "rec_ncu_shared_mem_layout" in ids


def test_warp_stall_sync_excludes_coalescing() -> None:
    ncu = _summary_with(stall="barrier", stall_pct=30.0, dram=25.0)
    result = _recs_for(ncu)

    # Coalescing is for memory bandwidth, not sync stalls
    assert "rec_ncu_improve_coalescing" not in result["ids"]


# ── Low issue efficiency ──────────────────────────────────────────────────────

def test_low_issue_efficiency_recommends_block_size_and_ilp() -> None:
    ncu = _summary_with(isu=35.0, stall="short_scoreboard", stall_pct=15.0)
    result = _recs_for(ncu)

    assert "low_issue_efficiency" in result["bottlenecks"]
    ids = result["ids"]
    assert "rec_ncu_increase_block_size" in ids
    assert "rec_ncu_instruction_level_parallelism" in ids


def test_low_issue_efficiency_suppressed_when_memory_stall_dominates() -> None:
    # Memory stalls dominate — low_issue_efficiency rule has suppressed_if warp_stall_is_memory
    ncu = _summary_with(isu=35.0, stall="memory_throttle", stall_pct=42.0, dram=80.0, mem_frac=0.8)
    result = _recs_for(ncu)

    ids = result["ids"]
    # ILP recs should NOT fire when memory stall dominates (the rule is suppressed)
    assert "rec_ncu_instruction_level_parallelism" not in ids


# ── Cache thrashing ───────────────────────────────────────────────────────────

def test_cache_thrashing_recommends_tiling() -> None:
    ncu = _summary_with(l1=25.0, l2=45.0, dram=50.0)
    result = _recs_for(ncu)

    assert "cache_thrashing" in result["bottlenecks"]
    assert "rec_ncu_tiling_shared_mem" in result["ids"]


# ── Insufficient NCU data ─────────────────────────────────────────────────────

def test_insufficient_ncu_data_recommends_collect_metrics() -> None:
    ncu = at.derive_ncu_run_summary([])
    result = _recs_for(ncu)

    assert result["bottlenecks"] == ["insufficient_ncu_data"]
    assert "rec_ncu_collect_metrics" in result["ids"]


# ── Full pipeline smoke test ──────────────────────────────────────────────────

def test_full_pipeline_scores_and_priorities_are_valid() -> None:
    text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "gemm,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,78.0",
        "gemm,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,%,7.0",
        "gemm,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,35.0",
        "gemm,l1tex__t_sector_hit_rate.pct,%,28.0",
        "gemm,sm__issue_active.avg.pct_of_peak_sustained_active,%,55.0",
    ])
    result = at.analyze_ncu_csv_text(text, environment={"mixed_precision": False, "num_gpus": 4})

    for rec in result["recommendations"]:
        assert 0.0 <= rec["score"] <= 1.0
        assert rec["priority"] in {"high", "medium", "low"}
        assert rec["tier"] in {"try_now", "next", "advanced"}
        assert isinstance(rec["actions"], list)
        assert isinstance(rec["validation"], list)
