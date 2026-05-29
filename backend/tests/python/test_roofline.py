"""Tests for roofline.py and its integration with signals/reconciliation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import pytest
from fournex.roofline import compute_roofline
from fournex.arch_profiles import get_arch_profile, _PROFILES


# ── Fixtures ─────────────────────────────────────────────────────────────────

H100_PROFILE = get_arch_profile("h100")
RTX5090_PROFILE = get_arch_profile("rtx5090")
A100_PROFILE = get_arch_profile("a100")


def _ncu_summary(*, dram_pct=None, tc_pct=None, isu_pct=None, roofline=None):
    return {
        "avg_dram_throughput_pct": dram_pct,
        "avg_tensor_core_utilization_pct": tc_pct,
        "avg_issue_slot_utilization_pct": isu_pct,
        **({"roofline": roofline} if roofline is not None else {}),
    }


# ── arch_profiles: peak specs present for all SM versions ────────────────────

@pytest.mark.parametrize("sm", list(_PROFILES.keys()))
def test_arch_profile_has_peak_fp32(sm):
    assert "peak_fp32_tflops" in _PROFILES[sm]
    assert isinstance(_PROFILES[sm]["peak_fp32_tflops"], float)
    assert _PROFILES[sm]["peak_fp32_tflops"] > 0


@pytest.mark.parametrize("sm", list(_PROFILES.keys()))
def test_arch_profile_has_peak_fp16(sm):
    assert "peak_fp16_tflops" in _PROFILES[sm]
    assert isinstance(_PROFILES[sm]["peak_fp16_tflops"], float)
    assert _PROFILES[sm]["peak_fp16_tflops"] > _PROFILES[sm]["peak_fp32_tflops"]


@pytest.mark.parametrize("sm", list(_PROFILES.keys()))
def test_arch_profile_has_peak_bw(sm):
    assert "peak_memory_bw_gbps" in _PROFILES[sm]
    assert isinstance(_PROFILES[sm]["peak_memory_bw_gbps"], float)
    assert _PROFILES[sm]["peak_memory_bw_gbps"] > 0


# ── compute_roofline: returns None on missing data ────────────────────────────

def test_returns_none_when_no_arch_specs():
    result = compute_roofline(_ncu_summary(dram_pct=80), {})
    assert result is None


def test_returns_none_when_arch_missing_bw():
    result = compute_roofline(_ncu_summary(dram_pct=80), {"peak_fp32_tflops": 19.5})
    assert result is None


def test_returns_none_when_all_utilization_zero():
    result = compute_roofline(_ncu_summary(dram_pct=0, tc_pct=0, isu_pct=0), H100_PROFILE)
    assert result is None


def test_returns_none_when_all_utilization_none():
    result = compute_roofline(_ncu_summary(), H100_PROFILE)
    assert result is None


# ── compute_roofline: basic output shape ─────────────────────────────────────

def test_result_has_all_required_keys():
    result = compute_roofline(_ncu_summary(dram_pct=80, isu_pct=60), H100_PROFILE)
    assert result is not None
    for key in (
        "arithmetic_intensity", "achieved_tflops", "peak_tflops",
        "peak_bw_gbps", "mfu_pct", "roofline_region",
        "memory_utilization_pct", "roofline_ceiling_tflops", "estimated",
    ):
        assert key in result, f"missing key: {key}"


def test_estimated_is_always_true():
    result = compute_roofline(_ncu_summary(dram_pct=50, isu_pct=70), A100_PROFILE)
    assert result["estimated"] is True


# ── Memory-bound classification ───────────────────────────────────────────────

def test_high_dram_low_compute_is_memory_bound():
    # H100: ridge ≈ 67 / 3350 * 1000 = 20 FLOPs/byte
    # dram_pct=90 → 0.9*3350=3015 GB/s, isu_pct=30 → 0.3*67=20.1 TFLOP/s
    # AI = 20.1 / 3015 * 1000 ≈ 6.7 FLOPs/byte < 20 → memory_bound
    result = compute_roofline(_ncu_summary(dram_pct=90, isu_pct=30), H100_PROFILE)
    assert result is not None
    assert result["roofline_region"] == "memory_bound"


def test_memory_bound_arithmetic_intensity_below_ridge():
    result = compute_roofline(_ncu_summary(dram_pct=90, isu_pct=30), H100_PROFILE)
    ridge = H100_PROFILE["peak_fp32_tflops"] / H100_PROFILE["peak_memory_bw_gbps"] * 1000
    assert result["arithmetic_intensity"] < ridge


def test_memory_utilization_matches_dram_pct():
    result = compute_roofline(_ncu_summary(dram_pct=75.5, isu_pct=40), H100_PROFILE)
    assert result["memory_utilization_pct"] == 75.5


# ── Compute-bound classification ─────────────────────────────────────────────

def test_high_compute_low_dram_is_compute_bound():
    # H100: dram=10% → 335 GB/s, isu=95% → 63.65 TFLOP/s
    # AI = 63.65 / 335 * 1000 ≈ 190 FLOPs/byte > 20 → compute_bound
    result = compute_roofline(_ncu_summary(dram_pct=10, isu_pct=95), H100_PROFILE)
    assert result is not None
    assert result["roofline_region"] == "compute_bound"


def test_zero_dram_is_compute_bound():
    result = compute_roofline(_ncu_summary(dram_pct=0, isu_pct=80), H100_PROFILE)
    assert result is not None
    assert result["roofline_region"] == "compute_bound"
    assert result["arithmetic_intensity"] is None


def test_zero_dram_ceiling_is_peak_tflops():
    result = compute_roofline(_ncu_summary(dram_pct=0, isu_pct=80), H100_PROFILE)
    assert result["roofline_ceiling_tflops"] == H100_PROFILE["peak_fp32_tflops"]


# ── TC path ───────────────────────────────────────────────────────────────────

def test_tc_dominated_uses_fp16_peak():
    # tc_pct=60 → uses fp16 path; peak_tflops should equal peak_fp16
    result = compute_roofline(_ncu_summary(dram_pct=30, tc_pct=60), H100_PROFILE)
    assert result is not None
    assert result["peak_tflops"] == H100_PROFILE["peak_fp16_tflops"]


def test_tc_below_threshold_uses_fp32_path():
    # tc_pct=3 < 5 threshold → uses FP32 path
    result = compute_roofline(_ncu_summary(dram_pct=50, tc_pct=3, isu_pct=70), H100_PROFILE)
    assert result is not None
    assert result["peak_tflops"] == H100_PROFILE["peak_fp32_tflops"]


def test_mfu_over_100_possible_with_tc():
    # TC path with high utilisation can exceed peak_fp32
    result = compute_roofline(_ncu_summary(dram_pct=5, tc_pct=90), H100_PROFILE)
    assert result is not None
    # achieved = 0.9 * 989 = 890.1; peak_fp32 = 67 → mfu_pct = 890.1/67*100 >> 100
    assert result["mfu_pct"] > 100.0


# ── MFU / achieved TFLOP/s ────────────────────────────────────────────────────

def test_mfu_proportional_to_isu():
    r50 = compute_roofline(_ncu_summary(dram_pct=0, isu_pct=50), H100_PROFILE)
    r100 = compute_roofline(_ncu_summary(dram_pct=0, isu_pct=100), H100_PROFILE)
    assert r50 is not None and r100 is not None
    assert abs(r100["mfu_pct"] / r50["mfu_pct"] - 2.0) < 0.01


def test_achieved_tflops_equals_isu_times_peak_fp32():
    result = compute_roofline(_ncu_summary(dram_pct=0, isu_pct=75), H100_PROFILE)
    expected = 0.75 * H100_PROFILE["peak_fp32_tflops"]
    assert abs(result["achieved_tflops"] - expected) < 0.01


# ── Roofline ceiling ──────────────────────────────────────────────────────────

def test_roofline_ceiling_is_min_of_bw_and_compute():
    # Memory-bound case: ceiling = bw_bound < compute ceiling
    result = compute_roofline(_ncu_summary(dram_pct=90, isu_pct=20), H100_PROFILE)
    ai = result["arithmetic_intensity"]
    expected_bw_ceil = H100_PROFILE["peak_memory_bw_gbps"] * ai / 1000
    assert result["roofline_ceiling_tflops"] <= H100_PROFILE["peak_fp32_tflops"]
    assert abs(result["roofline_ceiling_tflops"] - expected_bw_ceil) < 0.01


# ── RTX 5060/5090 (sm_120) validation ────────────────────────────────────────

def test_sm120_profile_has_reasonable_specs():
    p = _PROFILES["sm_120"]
    # RTX 5090: 104.8 TFLOP/s FP32, 1792 GB/s BW
    assert 50 < p["peak_fp32_tflops"] < 200
    assert 500 < p["peak_memory_bw_gbps"] < 5000


def test_rtx5090_roofline_computes():
    result = compute_roofline(_ncu_summary(dram_pct=50, isu_pct=60), RTX5090_PROFILE)
    assert result is not None
    assert result["roofline_region"] in ("memory_bound", "compute_bound")


# ── signals integration ───────────────────────────────────────────────────────

def test_analyze_ncu_uses_gpu_model_alias_for_arch_profile():
    from fournex.ncu_analysis import analyze_ncu_csv_text

    csv_text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "k,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,10.0",
        "k,sm__issue_active.avg.pct_of_peak_sustained_active,%,50.0",
    ])
    result = analyze_ncu_csv_text(csv_text, environment={"gpu_model": "h100"})
    roofline = result["ncu_run_summary"]["roofline"]
    assert roofline["peak_tflops"] == H100_PROFILE["peak_fp32_tflops"]


def test_analyze_ncu_applies_arch_profile_overrides():
    from fournex.ncu_analysis import analyze_ncu_csv_text

    csv_text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "k,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,10.0",
        "k,sm__issue_active.avg.pct_of_peak_sustained_active,%,50.0",
    ])
    result = analyze_ncu_csv_text(
        csv_text,
        environment={
            "gpu_model": "h100",
            "arch_profile_overrides": {"profiles": {"h100": {"peak_fp32_tflops": 60.0}}},
        },
    )
    assert result["ncu_run_summary"]["roofline"]["peak_tflops"] == 60.0


def test_extract_ncu_signals_exposes_roofline_keys():
    from fournex.recommendations.signals import extract_ncu_signals

    roofline_data = compute_roofline(_ncu_summary(dram_pct=80, isu_pct=50), H100_PROFILE)
    summary = _ncu_summary(dram_pct=80, isu_pct=50, roofline=roofline_data)
    signals = extract_ncu_signals(summary, [])

    assert "roofline_region" in signals
    assert "mfu_pct" in signals
    assert "arithmetic_intensity" in signals
    assert "roofline_estimated" in signals


def test_signals_roofline_region_matches_compute_roofline():
    from fournex.recommendations.signals import extract_ncu_signals

    roofline_data = compute_roofline(_ncu_summary(dram_pct=90, isu_pct=30), H100_PROFILE)
    summary = _ncu_summary(dram_pct=90, isu_pct=30, roofline=roofline_data)
    signals = extract_ncu_signals(summary, [])

    assert signals["roofline_region"] == "memory_bound"


def test_signals_without_roofline_returns_none_keys():
    from fournex.recommendations.signals import extract_ncu_signals

    summary = _ncu_summary(dram_pct=80, isu_pct=50)  # no "roofline" key
    signals = extract_ncu_signals(summary, [])

    assert signals["roofline_region"] is None
    assert signals["mfu_pct"] is None
    assert signals["roofline_estimated"] is False


# ── reconciliation integration ────────────────────────────────────────────────

def _ncu_result_with_roofline(roofline_region, mfu_pct=30.0):
    """Minimal ncu result dict that reconcile_evidence() can consume."""
    roofline = {
        "roofline_region": roofline_region,
        "mfu_pct": mfu_pct,
        "arithmetic_intensity": 5.0,
        "achieved_tflops": 10.0,
        "peak_tflops": 67.0,
        "peak_bw_gbps": 3350.0,
        "memory_utilization_pct": 80.0,
        "roofline_ceiling_tflops": 16.75,
        "estimated": True,
    }
    return {
        "schema": "ncu_analysis_v1",
        "ncu_run_summary": {
            "avg_dram_throughput_pct": 80.0,
            "avg_tensor_core_utilization_pct": None,
            "avg_issue_slot_utilization_pct": 30.0,
            "kernels_with_ncu_data": 1,
            "roofline": roofline,
        },
        "bottlenecks": [],
        "kernel_count": 1,
        "kernels_with_ncu_data": 1,
    }


def test_reconcile_detects_roofline_memory_bound():
    from fournex.reconciliation import reconcile_evidence

    ncu = _ncu_result_with_roofline("memory_bound", mfu_pct=15.0)
    result = reconcile_evidence(ncu=ncu)
    labels = [d["label"] for d in result["diagnoses"]]
    assert "roofline_memory_bound" in labels


def test_reconcile_roofline_memory_bound_not_fired_when_compute_bound():
    from fournex.reconciliation import reconcile_evidence

    ncu = _ncu_result_with_roofline("compute_bound", mfu_pct=80.0)
    result = reconcile_evidence(ncu=ncu)
    labels = [d["label"] for d in result["diagnoses"]]
    assert "roofline_memory_bound" not in labels


def test_reconcile_detects_roofline_low_mfu():
    from fournex.reconciliation import reconcile_evidence

    ncu = _ncu_result_with_roofline("compute_bound", mfu_pct=10.0)
    result = reconcile_evidence(ncu=ncu)
    labels = [d["label"] for d in result["diagnoses"]]
    assert "roofline_low_mfu" in labels


def test_reconcile_low_mfu_not_fired_when_memory_bound():
    # memory-bound kernel with low MFU should NOT fire roofline_low_mfu
    from fournex.reconciliation import reconcile_evidence

    ncu = _ncu_result_with_roofline("memory_bound", mfu_pct=10.0)
    result = reconcile_evidence(ncu=ncu)
    labels = [d["label"] for d in result["diagnoses"]]
    assert "roofline_low_mfu" not in labels


def test_reconcile_low_mfu_not_fired_when_mfu_above_threshold():
    from fournex.reconciliation import reconcile_evidence

    ncu = _ncu_result_with_roofline("compute_bound", mfu_pct=25.0)
    result = reconcile_evidence(ncu=ncu)
    labels = [d["label"] for d in result["diagnoses"]]
    assert "roofline_low_mfu" not in labels


def test_reconcile_roofline_memory_bound_has_ncu_metrics_in_evidence():
    from fournex.reconciliation import reconcile_evidence

    ncu = _ncu_result_with_roofline("memory_bound", mfu_pct=15.0)
    result = reconcile_evidence(ncu=ncu)
    diag = next(d for d in result["diagnoses"] if d["label"] == "roofline_memory_bound")
    assert diag["evidence"]["ncu"] is not None
    assert "metrics" in diag["evidence"]["ncu"]


def test_reconcile_roofline_diagnosis_has_recommendation_ids():
    from fournex.reconciliation import reconcile_evidence

    ncu = _ncu_result_with_roofline("memory_bound", mfu_pct=15.0)
    result = reconcile_evidence(ncu=ncu)
    diag = next(d for d in result["diagnoses"] if d["label"] == "roofline_memory_bound")
    assert "rec_roofline_tiling" in diag["recommendation_ids"]
    assert "rec_roofline_kernel_fusion" in diag["recommendation_ids"]
