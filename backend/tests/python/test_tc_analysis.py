"""Tests for tc_analysis.py and its integration with kernel_attribution / ncu_analysis."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import pytest
from fournex.tc_analysis import analyze_tc_efficiency, summarize_tc_analysis
from fournex.kernel_inspector import KernelLaunchSummary
from fournex.arch_profiles import get_arch_profile

H100 = get_arch_profile("h100")
A100 = get_arch_profile("a100")
NO_TC: dict = {}   # arch with no tensor_core_min_dim


# ── Helpers ───────────────────────────────────────────────────────────────────

def _summary(
    name: str = "k",
    tc: float | None = None,
    isu: float | None = None,
    dram: float | None = None,
) -> KernelLaunchSummary:
    s = KernelLaunchSummary(kernel_name=name)
    s.tensor_core_utilization_pct = tc
    s.issue_slot_utilization_pct = isu
    s.dram_throughput_pct = dram
    return s


def _analyze(tc=None, isu=None, dram=None, arch=None, env=None):
    arch = arch if arch is not None else H100
    return analyze_tc_efficiency(_summary(tc=tc, isu=isu, dram=dram), arch, env)


# ── Output schema ─────────────────────────────────────────────────────────────

def test_result_has_required_keys():
    r = _analyze(tc=60, isu=50)
    for key in (
        "tc_active", "tc_eligible", "tc_utilization_pct",
        "fallback_to_cuda_cores", "mixed_precision_active",
        "mixed_precision_opportunity", "efficiency_label", "diagnoses",
    ):
        assert key in r, f"missing key: {key}"


def test_diagnoses_is_list():
    r = _analyze(tc=60, isu=50)
    assert isinstance(r["diagnoses"], list)


# ── tc_active flag ────────────────────────────────────────────────────────────

def test_tc_active_true_above_threshold():
    assert _analyze(tc=6.0)["tc_active"] is True


def test_tc_active_false_at_threshold():
    # exactly at 5.0 is not above, so False
    assert _analyze(tc=5.0)["tc_active"] is False


def test_tc_active_false_below_threshold():
    assert _analyze(tc=3.0)["tc_active"] is False


def test_tc_active_false_when_none():
    assert _analyze(tc=None)["tc_active"] is False


# ── tc_eligible flag ──────────────────────────────────────────────────────────

def test_tc_eligible_true_compute_intensive_with_tc_arch():
    # isu > 20 → compute-intensive
    assert _analyze(tc=None, isu=25)["tc_eligible"] is True


def test_tc_eligible_true_via_tc_pct_proxy():
    # tc > 1.0 also counts even if isu is below threshold
    assert _analyze(tc=2.0, isu=10)["tc_eligible"] is True


def test_tc_eligible_false_on_no_tc_arch():
    r = analyze_tc_efficiency(_summary(tc=None, isu=30), NO_TC)
    assert r["tc_eligible"] is False


def test_tc_eligible_false_low_utilization_no_tc_proxy():
    # isu <= 20, tc <= 1.0 or None → not compute-intensive
    r = _analyze(tc=0.5, isu=15)
    assert r["tc_eligible"] is False


# ── fallback_to_cuda_cores ────────────────────────────────────────────────────

def test_fallback_detected_when_eligible_not_active_high_isu():
    r = _analyze(tc=2.0, isu=30)   # tc <= 5 (not active), isu > 20
    assert r["fallback_to_cuda_cores"] is True


def test_no_fallback_when_tc_active():
    r = _analyze(tc=40.0, isu=30)
    assert r["fallback_to_cuda_cores"] is False


def test_no_fallback_when_not_eligible():
    r = analyze_tc_efficiency(_summary(tc=2.0, isu=30), NO_TC)
    assert r["fallback_to_cuda_cores"] is False


def test_no_fallback_when_isu_low():
    # isu not > 20 → not compute-intensive enough to call it a fallback
    r = _analyze(tc=2.0, isu=10)
    assert r["fallback_to_cuda_cores"] is False


# ── mixed_precision_active ────────────────────────────────────────────────────

def test_mixed_precision_active_above_threshold():
    assert _analyze(tc=11.0)["mixed_precision_active"] is True


def test_mixed_precision_not_active_at_threshold():
    assert _analyze(tc=10.0)["mixed_precision_active"] is False


def test_mixed_precision_not_active_when_tc_none():
    assert _analyze(tc=None)["mixed_precision_active"] is False


# ── mixed_precision_opportunity ───────────────────────────────────────────────

def test_mp_opportunity_when_eligible_idle_bf16_not_on():
    # H100 supports bf16, tc idle, no env
    r = _analyze(tc=2.0, isu=30)
    assert r["mixed_precision_opportunity"] is True


def test_no_mp_opportunity_when_mp_already_enabled():
    r = _analyze(tc=2.0, isu=30, env={"mixed_precision": True})
    assert r["mixed_precision_opportunity"] is False


def test_no_mp_opportunity_when_tc_active():
    # If mixed precision is already active (tc > 10), no opportunity
    r = _analyze(tc=20.0, isu=30)
    assert r["mixed_precision_opportunity"] is False


def test_no_mp_opportunity_on_arch_without_bf16_or_fp8():
    # Arch with TC but no bf16 / fp8
    arch = {**H100}
    arch.pop("bf16_supported", None)
    arch.pop("fp8_supported", None)
    r = analyze_tc_efficiency(_summary(tc=2.0, isu=30), arch)
    assert r["mixed_precision_opportunity"] is False


# ── efficiency_label ──────────────────────────────────────────────────────────

def test_efficiency_label_efficient():
    assert _analyze(tc=60.0, isu=50)["efficiency_label"] == "efficient"


def test_efficiency_label_underutilized():
    # tc active (> 5) but <= 50
    assert _analyze(tc=20.0, isu=50)["efficiency_label"] == "underutilized"


def test_efficiency_label_inactive_when_eligible_not_active():
    # tc_eligible but tc not active
    assert _analyze(tc=3.0, isu=25)["efficiency_label"] == "inactive"


def test_efficiency_label_no_data_when_tc_none():
    assert _analyze(tc=None, isu=10)["efficiency_label"] == "no_data"


def test_efficiency_label_no_data_when_not_eligible_and_tc_low():
    # Not eligible + tc not active → no_data
    r = analyze_tc_efficiency(_summary(tc=2.0, isu=10), NO_TC)
    assert r["efficiency_label"] == "no_data"


# ── diagnoses content ─────────────────────────────────────────────────────────

def test_no_diagnoses_on_no_tc_arch():
    r = analyze_tc_efficiency(_summary(tc=40.0, isu=30), NO_TC)
    assert r["diagnoses"] == []


def test_fallback_diagnosis_present():
    r = _analyze(tc=2.0, isu=30)
    assert any("fallback" in d.lower() or "cuda core" in d.lower() for d in r["diagnoses"])


def test_mp_opportunity_diagnosis_present():
    r = _analyze(tc=2.0, isu=30)
    assert any("mixed" in d.lower() or "precision" in d.lower() for d in r["diagnoses"])


def test_underutilized_diagnosis_present():
    r = _analyze(tc=20.0, isu=30)
    assert any("tensor core" in d.lower() for d in r["diagnoses"])


def test_well_utilized_diagnosis_present():
    r = _analyze(tc=60.0, isu=50)
    assert any("well" in d.lower() or "tensor core" in d.lower() for d in r["diagnoses"])


def test_missing_metric_diagnosis_when_tc_none_and_arch_has_tc():
    r = _analyze(tc=None, isu=30)
    assert any("not available" in d.lower() or "metric" in d.lower() for d in r["diagnoses"])


# ── summarize_tc_analysis ──────────────────────────────────────────────────────

def test_summarize_empty_returns_defaults():
    r = summarize_tc_analysis([], H100)
    assert r["kernels_with_tc_data"] == 0
    assert r["kernels_tc_active"] == 0
    assert r["overall_efficiency_label"] == "no_data"
    assert r["top_finding"] is None


def test_summarize_schema():
    per = [_analyze(tc=60, isu=50), _analyze(tc=3, isu=30)]
    r = summarize_tc_analysis(per, H100)
    for key in (
        "kernels_with_tc_data", "kernels_tc_active", "kernels_tc_eligible_inactive",
        "kernels_fallback_to_cuda_cores", "avg_tc_utilization_pct",
        "any_mixed_precision_opportunity", "overall_efficiency_label", "top_finding",
    ):
        assert key in r, f"missing key: {key}"


def test_summarize_counts_active_kernels():
    per = [_analyze(tc=60), _analyze(tc=2), _analyze(tc=None)]
    r = summarize_tc_analysis(per, H100)
    assert r["kernels_tc_active"] == 1


def test_summarize_avg_tc_pct():
    per = [_analyze(tc=40), _analyze(tc=60)]
    r = summarize_tc_analysis(per, H100)
    assert r["avg_tc_utilization_pct"] == 50.0


def test_summarize_avg_none_when_no_data():
    per = [_analyze(tc=None), _analyze(tc=None)]
    r = summarize_tc_analysis(per, H100)
    assert r["avg_tc_utilization_pct"] is None


def test_summarize_overall_label_efficient():
    per = [_analyze(tc=60, isu=50), _analyze(tc=70, isu=60)]
    r = summarize_tc_analysis(per, H100)
    assert r["overall_efficiency_label"] == "efficient"


def test_summarize_overall_label_inactive_when_fallback():
    per = [_analyze(tc=2, isu=30)]
    r = summarize_tc_analysis(per, H100)
    assert r["overall_efficiency_label"] == "inactive"


def test_summarize_top_finding_not_none_for_fallback():
    per = [_analyze(tc=2, isu=30)]
    r = summarize_tc_analysis(per, H100)
    assert r["top_finding"] is not None


def test_summarize_top_finding_healthy():
    per = [_analyze(tc=60, isu=50), _analyze(tc=70, isu=60)]
    r = summarize_tc_analysis(per, H100)
    assert r["top_finding"] is not None
    assert "performing" in r["top_finding"].lower() or "tensor" in r["top_finding"].lower()


# ── Integration: tc_analysis embedded per-kernel in kernel_attribution ────────

def test_tc_analysis_embedded_in_kernel_attribution():
    from fournex.kernel_attribution import compute_kernel_attribution
    s = _summary("k", tc=60, isu=50)
    result = compute_kernel_attribution([s], H100)
    k = result["kernels"][0]
    assert "tc_analysis" in k
    assert k["tc_analysis"]["tc_active"] is True


def test_tc_analysis_keys_in_kernel_attribution():
    from fournex.kernel_attribution import compute_kernel_attribution
    s = _summary("k", tc=20, isu=30)
    result = compute_kernel_attribution([s], H100)
    tc = result["kernels"][0]["tc_analysis"]
    for key in ("tc_active", "tc_eligible", "tc_utilization_pct", "efficiency_label", "diagnoses"):
        assert key in tc


# ── Integration: tc_summary in ncu_analysis result ───────────────────────────

def test_ncu_result_has_tc_summary():
    from fournex.ncu_analysis import analyze_ncu_csv_text
    csv = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "k,dram__throughput_avg_pct_of_peak_sustained_elapsed,pct,80.0",
        "k,sm__issue_active_avg_pct_of_peak_sustained_active,pct,45.0",
        "k,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,pct,60.0",
    ])
    result = analyze_ncu_csv_text(csv, environment={"gpu_type": "h100"})
    assert "tc_summary" in result
    assert result["tc_summary"]["kernels_tc_active"] == 1
