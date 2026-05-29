"""Tests for kernel_attribution.py and its integration with ncu_analysis."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import pytest
from fournex.kernel_attribution import compute_kernel_attribution, _opportunity_score, _opportunity_label
from fournex.kernel_inspector import KernelLaunchSummary
from fournex.arch_profiles import get_arch_profile

H100 = get_arch_profile("h100")
A100 = get_arch_profile("a100")
EMPTY_PROFILE: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _summary(
    name: str = "kernel",
    dram: float | None = None,
    tc: float | None = None,
    isu: float | None = None,
    occ: float | None = None,
    duration_us: float | None = None,
) -> KernelLaunchSummary:
    s = KernelLaunchSummary(kernel_name=name)
    s.dram_throughput_pct = dram
    s.tensor_core_utilization_pct = tc
    s.issue_slot_utilization_pct = isu
    s.achieved_occupancy_pct = occ
    s.kernel_duration_us = duration_us
    return s


# ── Empty input ───────────────────────────────────────────────────────────────

def test_empty_summaries_returns_empty():
    result = compute_kernel_attribution([], H100)
    assert result["kernels"] == []
    assert result["top_opportunities"] == []
    assert result["has_runtime_share"] is False
    assert result["total_profiled_kernels"] == 0


# ── Output schema ─────────────────────────────────────────────────────────────

def test_result_has_required_keys():
    result = compute_kernel_attribution([_summary("k", dram=70, isu=40)], H100)
    for key in ("kernels", "top_opportunities", "has_runtime_share", "total_profiled_kernels"):
        assert key in result


def test_kernel_entry_has_required_fields():
    result = compute_kernel_attribution([_summary("k", dram=70, isu=40)], H100)
    k = result["kernels"][0]
    for field in (
        "kernel_name", "duration_us", "runtime_share_pct",
        "dram_throughput_pct", "tensor_core_utilization_pct",
        "achieved_occupancy_pct", "issue_slot_utilization_pct",
        "mfu_pct", "arithmetic_intensity", "roofline_region",
        "opportunity_score", "opportunity",
    ):
        assert field in k, f"missing field: {field}"


def test_total_profiled_kernels_count():
    summaries = [_summary(f"k{i}", dram=60, isu=50) for i in range(7)]
    result = compute_kernel_attribution(summaries, H100)
    assert result["total_profiled_kernels"] == 7


def test_top_opportunities_at_most_5():
    summaries = [_summary(f"k{i}", dram=50, isu=40) for i in range(10)]
    result = compute_kernel_attribution(summaries, H100)
    assert len(result["top_opportunities"]) <= 5


def test_top_opportunities_equals_kernels_when_few():
    summaries = [_summary(f"k{i}", dram=50, isu=40) for i in range(3)]
    result = compute_kernel_attribution(summaries, H100)
    assert len(result["top_opportunities"]) == 3


# ── Roofline integration ──────────────────────────────────────────────────────

def test_memory_bound_kernel_gets_region():
    # H100: high dram, low isu → AI < ridge → memory_bound
    s = _summary("bad", dram=90, isu=25)
    result = compute_kernel_attribution([s], H100)
    assert result["kernels"][0]["roofline_region"] == "memory_bound"


def test_compute_bound_kernel_gets_region():
    # Low dram, high isu → AI > ridge → compute_bound
    s = _summary("fast", dram=10, isu=92)
    result = compute_kernel_attribution([s], H100)
    assert result["kernels"][0]["roofline_region"] == "compute_bound"


def test_no_arch_profile_roofline_fields_are_none():
    s = _summary("k", dram=80, isu=50)
    result = compute_kernel_attribution([s], EMPTY_PROFILE)
    k = result["kernels"][0]
    assert k["mfu_pct"] is None
    assert k["arithmetic_intensity"] is None
    assert k["roofline_region"] is None


def test_kernel_name_preserved():
    s = _summary("fused_attention_kernel", dram=85, isu=30)
    result = compute_kernel_attribution([s], H100)
    assert result["kernels"][0]["kernel_name"] == "fused_attention_kernel"


# ── Runtime share ─────────────────────────────────────────────────────────────

def test_has_runtime_share_false_when_no_duration():
    summaries = [_summary("k1", dram=70), _summary("k2", dram=50)]
    result = compute_kernel_attribution(summaries, H100)
    assert result["has_runtime_share"] is False
    for k in result["kernels"]:
        assert k["runtime_share_pct"] is None


def test_has_runtime_share_true_when_duration_present():
    summaries = [
        _summary("k1", dram=80, duration_us=100.0),
        _summary("k2", dram=50, duration_us=50.0),
    ]
    result = compute_kernel_attribution(summaries, H100)
    assert result["has_runtime_share"] is True


def test_runtime_share_sums_to_100():
    summaries = [
        _summary("k1", dram=80, duration_us=300.0),
        _summary("k2", dram=50, duration_us=100.0),
        _summary("k3", dram=30, duration_us=100.0),
    ]
    result = compute_kernel_attribution(summaries, H100)
    total = sum(k["runtime_share_pct"] for k in result["kernels"])
    assert abs(total - 100.0) < 0.01


def test_runtime_share_proportional_to_duration():
    summaries = [
        _summary("big", dram=80, duration_us=600.0),
        _summary("small", dram=80, duration_us=200.0),
    ]
    result = compute_kernel_attribution(summaries, H100)
    by_name = {k["kernel_name"]: k for k in result["kernels"]}
    assert abs(by_name["big"]["runtime_share_pct"] - 75.0) < 0.01
    assert abs(by_name["small"]["runtime_share_pct"] - 25.0) < 0.01


def test_partial_duration_data_handled():
    # One kernel has duration, one doesn't — has_runtime_share True, missing one gets None
    summaries = [
        _summary("k1", dram=80, duration_us=200.0),
        _summary("k2", dram=60),  # no duration
    ]
    result = compute_kernel_attribution(summaries, H100)
    assert result["has_runtime_share"] is True
    by_name = {k["kernel_name"]: k for k in result["kernels"]}
    assert by_name["k1"]["runtime_share_pct"] == 100.0
    assert by_name["k2"]["runtime_share_pct"] is None


# ── Opportunity scoring ───────────────────────────────────────────────────────

def test_sorted_by_opportunity_score_descending():
    summaries = [
        _summary("fast", dram=10, isu=90),        # low gap → low score
        _summary("bottleneck", dram=90, isu=20),   # memory-bound, high gap → high score
        _summary("mid", dram=55, isu=50),          # medium
    ]
    result = compute_kernel_attribution(summaries, H100)
    scores = [k["opportunity_score"] for k in result["kernels"]]
    assert scores == sorted(scores, reverse=True)


def test_memory_bound_scores_higher_than_compute_bound():
    # memory_bound kernel vs compute-bound well-utilized kernel
    mem_bound = _summary("mem", dram=90, isu=25)
    compute_ok = _summary("cmp", dram=10, isu=85)
    result = compute_kernel_attribution([mem_bound, compute_ok], H100)
    scores = {k["kernel_name"]: k["opportunity_score"] for k in result["kernels"]}
    assert scores["mem"] > scores["cmp"]


def test_high_runtime_share_boosts_score():
    # Same kernel profile, different runtime share — higher share = higher score
    slow = _summary("slow", dram=80, isu=30, duration_us=800.0)
    fast = _summary("fast", dram=80, isu=30, duration_us=200.0)
    result = compute_kernel_attribution([slow, fast], H100)
    scores = {k["kernel_name"]: k["opportunity_score"] for k in result["kernels"]}
    assert scores["slow"] > scores["fast"]


def test_well_utilized_kernel_low_score():
    # Near-peak kernel: low opportunity
    s = _summary("peak", dram=20, isu=95)
    result = compute_kernel_attribution([s], H100)
    k = result["kernels"][0]
    assert k["opportunity_score"] < 0.3
    assert k["opportunity"] in ("low", "medium")


def test_memory_bound_high_share_opportunity_high():
    # 70% of GPU time, memory-bound — should be "high"
    s = _summary("hot", dram=88, isu=30, duration_us=700.0)
    bystander = _summary("cold", dram=20, isu=80, duration_us=300.0)
    result = compute_kernel_attribution([s, bystander], H100)
    hot = next(k for k in result["kernels"] if k["kernel_name"] == "hot")
    assert hot["opportunity"] == "high"


def test_opportunity_label_low_for_zero_score():
    assert _opportunity_label(0.0, has_runtime_share=True) == "low"
    assert _opportunity_label(0.0, has_runtime_share=False) == "low"


def test_opportunity_label_high_thresholds_differ_by_runtime_mode():
    # With runtime share: >0.25 is high
    assert _opportunity_label(0.30, has_runtime_share=True) == "high"
    assert _opportunity_label(0.20, has_runtime_share=True) == "medium"
    # Without runtime share: >0.55 is high
    assert _opportunity_label(0.60, has_runtime_share=False) == "high"
    assert _opportunity_label(0.40, has_runtime_share=False) == "medium"


# ── kernel_duration_us extraction from NCU metric ────────────────────────────

def test_duration_alias_in_canonical_metric():
    from fournex.kernel_inspector import _canonical_ncu_metric_name
    assert _canonical_ncu_metric_name("duration") == "kernel_duration_us"
    assert _canonical_ncu_metric_name("Duration") == "kernel_duration_us"
    assert _canonical_ncu_metric_name("gpu__time_duration_sum") == "kernel_duration_us"


def test_duration_parsed_from_ncu_csv():
    from fournex.kernel_inspector import parse_nsight_compute_csv_text
    csv_text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "attn_kernel,duration,us,142.5",
        "attn_kernel,dram__throughput_avg_pct_of_peak_sustained_elapsed,pct,85.0",
    ])
    summaries = parse_nsight_compute_csv_text(csv_text)
    assert len(summaries) == 1
    assert summaries[0].kernel_duration_us == 142.5


def test_duration_none_when_not_in_csv():
    from fournex.kernel_inspector import parse_nsight_compute_csv_text
    csv_text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "k,dram__throughput_avg_pct_of_peak_sustained_elapsed,pct,70.0",
    ])
    summaries = parse_nsight_compute_csv_text(csv_text)
    assert summaries[0].kernel_duration_us is None


# ── Integration with analyze_ncu_csv_text ────────────────────────────────────

def test_ncu_result_has_kernel_attribution_key():
    from fournex.ncu_analysis import analyze_ncu_csv_text
    csv_text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "my_kernel,dram__throughput_avg_pct_of_peak_sustained_elapsed,pct,80.0",
        "my_kernel,sm__issue_active_avg_pct_of_peak_sustained_active,pct,45.0",
    ])
    result = analyze_ncu_csv_text(csv_text, environment={"gpu_type": "h100"})
    assert "kernel_attribution" in result


def test_ncu_result_attribution_has_kernels():
    from fournex.ncu_analysis import analyze_ncu_csv_text
    csv_text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "attn,dram__throughput_avg_pct_of_peak_sustained_elapsed,pct,88.0",
        "attn,sm__issue_active_avg_pct_of_peak_sustained_active,pct,30.0",
        "norm,dram__throughput_avg_pct_of_peak_sustained_elapsed,pct,40.0",
        "norm,sm__issue_active_avg_pct_of_peak_sustained_active,pct,75.0",
    ])
    result = analyze_ncu_csv_text(csv_text, environment={"gpu_type": "h100"})
    attr = result["kernel_attribution"]
    assert attr["total_profiled_kernels"] == 2
    names = [k["kernel_name"] for k in attr["kernels"]]
    assert "attn" in names
    assert "norm" in names


def test_ncu_result_attribution_sorted_worst_first():
    from fournex.ncu_analysis import analyze_ncu_csv_text
    csv_text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "fast_kernel,dram__throughput_avg_pct_of_peak_sustained_elapsed,pct,10.0",
        "fast_kernel,sm__issue_active_avg_pct_of_peak_sustained_active,pct,92.0",
        "slow_kernel,dram__throughput_avg_pct_of_peak_sustained_elapsed,pct,90.0",
        "slow_kernel,sm__issue_active_avg_pct_of_peak_sustained_active,pct,22.0",
    ])
    result = analyze_ncu_csv_text(csv_text, environment={"gpu_type": "h100"})
    kernels = result["kernel_attribution"]["kernels"]
    assert kernels[0]["kernel_name"] == "slow_kernel"
    assert kernels[-1]["kernel_name"] == "fast_kernel"
