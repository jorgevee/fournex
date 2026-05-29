"""Tests for occupancy_analysis.py and its integration with kernel_attribution / ncu_analysis."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import pytest
from fournex.occupancy_analysis import analyze_occupancy, summarize_occupancy_analysis
from fournex.kernel_inspector import KernelLaunchSummary
from fournex.arch_profiles import get_arch_profile

H100 = get_arch_profile("h100")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _summary(
    name: str = "k",
    achieved_occ: float | None = None,
    occupancy_estimate: dict | None = None,
    regs: int | None = None,
    shared: int | None = None,
    threads: int | None = None,
) -> KernelLaunchSummary:
    s = KernelLaunchSummary(kernel_name=name)
    s.achieved_occupancy_pct = achieved_occ
    s.occupancy_estimate = occupancy_estimate or {}
    s.registers_per_thread = regs
    s.shared_memory_per_block_bytes = shared
    s.threads_per_block = threads
    return s


def _estimate(
    occ_pct: float | None = None,
    limiters: list[str] | None = None,
    blocks_limits: dict | None = None,
) -> dict:
    return {
        "occupancy_pct": occ_pct,
        "limiting_factors": limiters or [],
        "blocks_per_sm_limits": blocks_limits or {},
    }


# ── Output schema ─────────────────────────────────────────────────────────────

def test_result_has_required_keys():
    s = _summary(achieved_occ=50.0, occupancy_estimate=_estimate(60.0, ["registers"]))
    r = analyze_occupancy(s)
    for key in (
        "achieved_occupancy_pct", "theoretical_occupancy_pct", "occupancy_gap_pct",
        "occupancy_efficiency_pct", "primary_limiter", "all_limiters",
        "blocks_per_sm_limits", "registers_per_thread",
        "shared_memory_per_block_bytes", "threads_per_block", "diagnosis",
    ):
        assert key in r, f"missing key: {key}"


# ── Passthrough of raw fields ─────────────────────────────────────────────────

def test_achieved_pct_passed_through():
    r = analyze_occupancy(_summary(achieved_occ=62.5))
    assert r["achieved_occupancy_pct"] == 62.5


def test_theoretical_pct_from_estimate():
    s = _summary(occupancy_estimate=_estimate(occ_pct=75.0))
    assert analyze_occupancy(s)["theoretical_occupancy_pct"] == 75.0


def test_registers_per_thread_passed_through():
    r = analyze_occupancy(_summary(regs=48))
    assert r["registers_per_thread"] == 48


def test_shared_memory_passed_through():
    r = analyze_occupancy(_summary(shared=16384))
    assert r["shared_memory_per_block_bytes"] == 16384


def test_threads_per_block_passed_through():
    r = analyze_occupancy(_summary(threads=256))
    assert r["threads_per_block"] == 256


# ── Gap and efficiency math ───────────────────────────────────────────────────

def test_gap_is_theoretical_minus_achieved():
    s = _summary(achieved_occ=50.0, occupancy_estimate=_estimate(occ_pct=80.0))
    r = analyze_occupancy(s)
    assert r["occupancy_gap_pct"] == pytest.approx(30.0, abs=0.01)


def test_gap_none_when_achieved_missing():
    s = _summary(occupancy_estimate=_estimate(occ_pct=80.0))
    assert analyze_occupancy(s)["occupancy_gap_pct"] is None


def test_gap_none_when_theoretical_missing():
    s = _summary(achieved_occ=50.0)
    assert analyze_occupancy(s)["occupancy_gap_pct"] is None


def test_efficiency_pct_calculation():
    s = _summary(achieved_occ=60.0, occupancy_estimate=_estimate(occ_pct=80.0))
    r = analyze_occupancy(s)
    assert r["occupancy_efficiency_pct"] == pytest.approx(75.0, abs=0.1)


def test_efficiency_none_when_theoretical_zero():
    s = _summary(achieved_occ=50.0, occupancy_estimate=_estimate(occ_pct=0.0))
    assert analyze_occupancy(s)["occupancy_efficiency_pct"] is None


def test_efficiency_none_when_achieved_missing():
    s = _summary(occupancy_estimate=_estimate(occ_pct=80.0))
    assert analyze_occupancy(s)["occupancy_efficiency_pct"] is None


# ── Limiters ──────────────────────────────────────────────────────────────────

def test_primary_limiter_is_first_factor():
    s = _summary(occupancy_estimate=_estimate(limiters=["registers", "shared_memory"]))
    r = analyze_occupancy(s)
    assert r["primary_limiter"] == "registers"


def test_all_limiters_preserved():
    factors = ["shared_memory", "blocks"]
    s = _summary(occupancy_estimate=_estimate(limiters=factors))
    assert analyze_occupancy(s)["all_limiters"] == factors


def test_primary_limiter_none_when_no_factors():
    s = _summary(occupancy_estimate=_estimate())
    assert analyze_occupancy(s)["primary_limiter"] is None


def test_blocks_per_sm_limits_preserved():
    limits = {"registers": 4, "shared_memory": 8}
    s = _summary(occupancy_estimate=_estimate(blocks_limits=limits))
    assert analyze_occupancy(s)["blocks_per_sm_limits"] == limits


# ── diagnosis text ────────────────────────────────────────────────────────────

def test_no_diagnosis_when_both_pct_missing():
    s = _summary()   # no achieved, no theoretical
    assert analyze_occupancy(s)["diagnosis"] is None


def test_register_limiter_diagnosis_mentions_register():
    s = _summary(
        achieved_occ=50.0,
        occupancy_estimate=_estimate(occ_pct=80.0, limiters=["registers"]),
        regs=64,
    )
    d = analyze_occupancy(s)["diagnosis"]
    assert d is not None
    assert "register" in d.lower()


def test_shared_memory_limiter_diagnosis_mentions_shared():
    s = _summary(
        achieved_occ=50.0,
        occupancy_estimate=_estimate(occ_pct=80.0, limiters=["shared_memory"]),
        shared=32768,
    )
    d = analyze_occupancy(s)["diagnosis"]
    assert d is not None
    assert "shared" in d.lower()


def test_diagnosis_includes_register_count():
    s = _summary(
        achieved_occ=50.0,
        occupancy_estimate=_estimate(occ_pct=80.0, limiters=["registers"]),
        regs=48,
    )
    d = analyze_occupancy(s)["diagnosis"]
    assert "48" in d


def test_diagnosis_includes_shared_memory_kb():
    s = _summary(
        achieved_occ=50.0,
        occupancy_estimate=_estimate(occ_pct=80.0, limiters=["shared_memory"]),
        shared=16384,
    )
    d = analyze_occupancy(s)["diagnosis"]
    assert "16.0 KB" in d or "16.0" in d


def test_diagnosis_low_efficiency_note():
    # Efficiency < 80% should note the gap
    s = _summary(
        achieved_occ=40.0,
        occupancy_estimate=_estimate(occ_pct=80.0, limiters=["registers"]),
    )
    d = analyze_occupancy(s)["diagnosis"]
    assert d is not None
    assert "below" in d.lower() or "ceiling" in d.lower()


def test_diagnosis_low_occ_fallback_without_limiter():
    # Low achieved but no known limiter still gets a diagnosis
    s = _summary(achieved_occ=20.0, occupancy_estimate=_estimate())
    d = analyze_occupancy(s)["diagnosis"]
    assert d is not None
    assert "20" in d


def test_diagnosis_none_when_occupancy_healthy_no_limiter():
    # High achieved, high efficiency, no limiter → nothing to say
    s = _summary(achieved_occ=85.0, occupancy_estimate=_estimate(occ_pct=90.0))
    d = analyze_occupancy(s)["diagnosis"]
    # efficiency = 85/90 ≈ 94% → no low-efficiency note; no limiter → no limiter msg
    # achieved >= 40 → no low-occ fallback
    assert d is None


# ── summarize_occupancy_analysis ──────────────────────────────────────────────

def test_summarize_empty_returns_defaults():
    r = summarize_occupancy_analysis([])
    assert r["kernels_with_occupancy_data"] == 0
    assert r["kernels_low_occupancy"] == 0
    assert r["avg_achieved_occupancy_pct"] is None
    assert r["dominant_limiter"] is None
    assert r["top_finding"] is None


def test_summarize_schema():
    per = [analyze_occupancy(_summary(achieved_occ=50.0, occupancy_estimate=_estimate(60.0, ["registers"])))]
    r = summarize_occupancy_analysis(per)
    for key in (
        "kernels_with_occupancy_data", "kernels_low_occupancy",
        "avg_achieved_occupancy_pct", "avg_occupancy_gap_pct",
        "avg_occupancy_efficiency_pct", "limiter_counts",
        "dominant_limiter", "top_finding",
    ):
        assert key in r, f"missing key: {key}"


def test_summarize_counts_kernels_with_data():
    per = [
        analyze_occupancy(_summary(achieved_occ=50.0)),
        analyze_occupancy(_summary(achieved_occ=None)),
    ]
    assert summarize_occupancy_analysis(per)["kernels_with_occupancy_data"] == 1


def test_summarize_low_occupancy_threshold():
    per = [
        analyze_occupancy(_summary(achieved_occ=20.0)),   # below 40%
        analyze_occupancy(_summary(achieved_occ=60.0)),   # above 40%
    ]
    assert summarize_occupancy_analysis(per)["kernels_low_occupancy"] == 1


def test_summarize_avg_achieved_pct():
    per = [
        analyze_occupancy(_summary(achieved_occ=40.0)),
        analyze_occupancy(_summary(achieved_occ=60.0)),
    ]
    r = summarize_occupancy_analysis(per)
    assert r["avg_achieved_occupancy_pct"] == pytest.approx(50.0, abs=0.01)


def test_summarize_avg_gap():
    per = [
        analyze_occupancy(_summary(achieved_occ=50.0, occupancy_estimate=_estimate(70.0))),
        analyze_occupancy(_summary(achieved_occ=60.0, occupancy_estimate=_estimate(80.0))),
    ]
    r = summarize_occupancy_analysis(per)
    # gaps: 20 and 20 → avg 20
    assert r["avg_occupancy_gap_pct"] == pytest.approx(20.0, abs=0.01)


def test_summarize_limiter_counts():
    per = [
        analyze_occupancy(_summary(occupancy_estimate=_estimate(limiters=["registers"]))),
        analyze_occupancy(_summary(occupancy_estimate=_estimate(limiters=["registers", "shared_memory"]))),
    ]
    r = summarize_occupancy_analysis(per)
    assert r["limiter_counts"]["registers"] == 2
    assert r["limiter_counts"]["shared_memory"] == 1


def test_summarize_dominant_limiter():
    per = [
        analyze_occupancy(_summary(occupancy_estimate=_estimate(limiters=["registers"]))),
        analyze_occupancy(_summary(occupancy_estimate=_estimate(limiters=["registers"]))),
        analyze_occupancy(_summary(occupancy_estimate=_estimate(limiters=["shared_memory"]))),
    ]
    assert summarize_occupancy_analysis(per)["dominant_limiter"] == "registers"


def test_summarize_top_finding_low_occ():
    per = [
        analyze_occupancy(_summary(achieved_occ=20.0, occupancy_estimate=_estimate(limiters=["registers"]))),
    ]
    r = summarize_occupancy_analysis(per)
    assert r["top_finding"] is not None
    assert "1/1" in r["top_finding"] or "occupancy" in r["top_finding"].lower()


def test_summarize_top_finding_healthy():
    per = [
        analyze_occupancy(_summary(achieved_occ=80.0, occupancy_estimate=_estimate(occ_pct=90.0))),
        analyze_occupancy(_summary(achieved_occ=75.0, occupancy_estimate=_estimate(occ_pct=85.0))),
    ]
    r = summarize_occupancy_analysis(per)
    # All above 40%, efficiency ≥ 80% → healthy finding
    assert r["top_finding"] is not None
    assert "healthy" in r["top_finding"].lower()


# ── Integration: occupancy_analysis embedded per-kernel ──────────────────────

def test_occupancy_analysis_embedded_in_kernel_attribution():
    from fournex.kernel_attribution import compute_kernel_attribution
    from fournex.kernel_inspector import KernelLaunchSummary
    s = KernelLaunchSummary(kernel_name="k")
    s.achieved_occupancy_pct = 55.0
    s.occupancy_estimate = _estimate(occ_pct=70.0, limiters=["registers"])
    s.registers_per_thread = 48
    result = compute_kernel_attribution([s], H100)
    k = result["kernels"][0]
    assert "occupancy_analysis" in k
    assert k["occupancy_analysis"]["achieved_occupancy_pct"] == 55.0


def test_occupancy_analysis_keys_in_kernel_attribution():
    from fournex.kernel_attribution import compute_kernel_attribution
    from fournex.kernel_inspector import KernelLaunchSummary
    s = KernelLaunchSummary(kernel_name="k")
    s.achieved_occupancy_pct = 50.0
    result = compute_kernel_attribution([s], H100)
    occ = result["kernels"][0]["occupancy_analysis"]
    for key in ("achieved_occupancy_pct", "primary_limiter", "diagnosis"):
        assert key in occ


# ── Integration: occupancy_summary in ncu_analysis result ────────────────────

def test_ncu_result_has_occupancy_summary():
    from fournex.ncu_analysis import analyze_ncu_csv_text
    csv = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "k,dram__throughput_avg_pct_of_peak_sustained_elapsed,pct,80.0",
        "k,sm__issue_active_avg_pct_of_peak_sustained_active,pct,45.0",
    ])
    result = analyze_ncu_csv_text(csv, environment={"gpu_type": "h100"})
    assert "occupancy_summary" in result
    assert "kernels_with_occupancy_data" in result["occupancy_summary"]
