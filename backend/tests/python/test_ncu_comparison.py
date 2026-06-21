import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as at


# ── CSV fixtures ──────────────────────────────────────────────────────────────

def _csv(*rows: str) -> str:
    return "\n".join(rows)


# Heavily memory-bound baseline: high DRAM, low cache, memory stalls, low TC
BASELINE_MEMORY_BOUND = _csv(
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,88.0",
    "ker,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,%,8.0",
    "ker,l1tex__t_sector_hit_rate.pct,%,28.0",
    "ker,lts__t_sector_hit_rate.pct,%,38.0",
    "ker,sm__issue_active.avg.pct_of_peak_sustained_active,%,55.0",
    "ker,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,45.0",
    "ker,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_long_scoreboard,%,22.0",
)

# Optimized: DRAM reduced, caches improved, memory stalls mostly gone
OPTIMIZED_MEMORY_RESOLVED = _csv(
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,42.0",
    "ker,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,%,62.0",
    "ker,l1tex__t_sector_hit_rate.pct,%,72.0",
    "ker,lts__t_sector_hit_rate.pct,%,80.0",
    "ker,sm__issue_active.avg.pct_of_peak_sustained_active,%,75.0",
    "ker,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,8.0",
    "ker,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_long_scoreboard,%,4.0",
)

# Regression: introduces a sync bottleneck the baseline didn't have
REGRESSED_NEW_SYNC = _csv(
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,40.0",
    "ker,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,%,62.0",
    "ker,l1tex__t_sector_hit_rate.pct,%,74.0",
    "ker,lts__t_sector_hit_rate.pct,%,82.0",
    "ker,sm__issue_active.avg.pct_of_peak_sustained_active,%,75.0",
    "ker,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_barrier,%,35.0",
    "ker,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_wait,%,25.0",
)

# Identical to baseline → neutral
BASELINE_LOW_ISSUE = _csv(
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,30.0",
    "ker,sm__issue_active.avg.pct_of_peak_sustained_active,%,38.0",
)

OPTIMIZED_SAME_LOW_ISSUE = _csv(
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,30.0",
    "ker,sm__issue_active.avg.pct_of_peak_sustained_active,%,38.0",
)


# ── Schema ────────────────────────────────────────────────────────────────────

def test_diff_ncu_runs_schema() -> None:
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, OPTIMIZED_MEMORY_RESOLVED)

    assert result["schema"] == "ncu_comparison_v1"
    assert "label_baseline" in result
    assert "label_optimized" in result
    assert "baseline" in result
    assert "optimized" in result
    assert "bottleneck_diff" in result
    assert "metric_deltas" in result
    assert "verdict" in result


def test_diff_ncu_runs_default_labels() -> None:
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, OPTIMIZED_MEMORY_RESOLVED)
    assert result["label_baseline"] == "baseline"
    assert result["label_optimized"] == "optimized"


def test_diff_ncu_runs_custom_labels() -> None:
    result = at.diff_ncu_runs(
        BASELINE_MEMORY_BOUND,
        OPTIMIZED_MEMORY_RESOLVED,
        label_baseline="v1",
        label_optimized="v2_tiled",
    )
    assert result["label_baseline"] == "v1"
    assert result["label_optimized"] == "v2_tiled"


# ── Bottleneck diff ───────────────────────────────────────────────────────────

def test_bottleneck_diff_resolved_bottlenecks() -> None:
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, OPTIMIZED_MEMORY_RESOLVED)
    bdiff = result["bottleneck_diff"]

    # memory-bound baseline → optimized removes those bottlenecks
    assert len(bdiff["resolved"]) > 0
    # No new bottlenecks introduced
    assert bdiff["new"] == []


def test_bottleneck_diff_new_regression() -> None:
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, REGRESSED_NEW_SYNC)
    bdiff = result["bottleneck_diff"]

    # warp_stall_sync introduced by barrier stalls
    assert "warp_stall_sync" in bdiff["new"]


def test_bottleneck_diff_neutral_same_input() -> None:
    result = at.diff_ncu_runs(BASELINE_LOW_ISSUE, OPTIMIZED_SAME_LOW_ISSUE)
    bdiff = result["bottleneck_diff"]

    assert bdiff["resolved"] == []
    assert bdiff["new"] == []


def test_bottleneck_diff_score_deltas_present() -> None:
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, OPTIMIZED_MEMORY_RESOLVED)
    bdiff = result["bottleneck_diff"]

    # score_deltas covers all persistent bottlenecks
    for label in bdiff["persistent"]:
        assert label in bdiff["score_deltas"]


# ── Metric deltas ─────────────────────────────────────────────────────────────

def test_metric_deltas_dram_improved() -> None:
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, OPTIMIZED_MEMORY_RESOLVED)
    deltas = result["metric_deltas"]

    assert "avg_dram_throughput_pct" in deltas
    d = deltas["avg_dram_throughput_pct"]
    assert d["baseline"] == 88.0
    assert d["optimized"] == 42.0
    assert d["delta"] < 0
    assert d["direction"] == "improved"  # lower DRAM is better


def test_metric_deltas_cache_improved() -> None:
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, OPTIMIZED_MEMORY_RESOLVED)
    deltas = result["metric_deltas"]

    assert deltas["avg_l1_cache_hit_rate_pct"]["direction"] == "improved"
    assert deltas["avg_l2_cache_hit_rate_pct"]["direction"] == "improved"


def test_metric_deltas_memory_stall_fraction_improved() -> None:
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, OPTIMIZED_MEMORY_RESOLVED)
    deltas = result["metric_deltas"]

    d = deltas["memory_stall_fraction"]
    # Baseline: (45+22)/100 = 0.67; Optimized: (8+4)/100 = 0.12 → improved
    assert d["direction"] == "improved"
    assert d["delta"] < 0


def test_metric_deltas_neutral_same_input() -> None:
    result = at.diff_ncu_runs(BASELINE_LOW_ISSUE, OPTIMIZED_SAME_LOW_ISSUE)
    deltas = result["metric_deltas"]

    d = deltas["avg_dram_throughput_pct"]
    assert d["direction"] == "neutral"
    assert d["delta"] == 0.0


# ── Verdict ───────────────────────────────────────────────────────────────────

def test_verdict_improved_outcome() -> None:
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, OPTIMIZED_MEMORY_RESOLVED)
    verdict = result["verdict"]

    assert verdict["outcome"] == "improved"
    assert verdict["bottlenecks_resolved"] > 0
    assert verdict["bottlenecks_new"] == 0


def test_verdict_regressed_outcome() -> None:
    # Baseline has almost no bottlenecks; regressed introduces sync stall
    clean_baseline = _csv(
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,40.0",
        "ker,l1tex__t_sector_hit_rate.pct,%,75.0",
        "ker,lts__t_sector_hit_rate.pct,%,82.0",
        "ker,sm__issue_active.avg.pct_of_peak_sustained_active,%,72.0",
        "ker,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_wait,%,3.0",
    )
    result = at.diff_ncu_runs(clean_baseline, REGRESSED_NEW_SYNC)
    verdict = result["verdict"]

    assert verdict["outcome"] == "regressed"
    assert verdict["bottlenecks_new"] > 0


def test_verdict_neutral_outcome() -> None:
    result = at.diff_ncu_runs(BASELINE_LOW_ISSUE, OPTIMIZED_SAME_LOW_ISSUE)
    assert result["verdict"]["outcome"] == "neutral"


def test_verdict_mixed_outcome() -> None:
    # Baseline has memory bottleneck; optimized resolves memory but introduces sync
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, REGRESSED_NEW_SYNC)
    verdict = result["verdict"]
    # memory bottlenecks resolved, sync introduced → mixed
    assert verdict["outcome"] == "mixed"


# ── Kernel GPU time (the trustworthy speedup basis) ───────────────────────────

BASELINE_WITH_DURATION = _csv(
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "ker,gpu__time_duration.sum,usecond,1000.0",
    "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,88.0",
    "ker,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld,sector,8.0",
)

OPTIMIZED_WITH_DURATION = _csv(
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "ker,gpu__time_duration.sum,usecond,250.0",
    "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,40.0",
    "ker,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld,sector,1.1",
)


def test_kernel_time_diff_present_and_available() -> None:
    result = at.diff_ncu_runs(BASELINE_WITH_DURATION, OPTIMIZED_WITH_DURATION)
    kt = result["kernel_time"]
    assert kt["available"] is True
    assert kt["baseline_us"] == 1000.0
    assert kt["optimized_us"] == 250.0
    assert kt["speedup_x"] == pytest.approx(4.0)


def test_verdict_basis_is_kernel_time_when_duration_present() -> None:
    result = at.diff_ncu_runs(BASELINE_WITH_DURATION, OPTIMIZED_WITH_DURATION)
    v = result["verdict"]
    assert v["basis"] == "kernel_gpu_time"
    assert v["outcome"] == "improved"
    assert v["kernel_speedup_x"] == pytest.approx(4.0)


def test_verdict_regressed_from_kernel_time() -> None:
    # Reverse direction: the "optimized" run is now slower on the GPU.
    result = at.diff_ncu_runs(OPTIMIZED_WITH_DURATION, BASELINE_WITH_DURATION)
    v = result["verdict"]
    assert v["basis"] == "kernel_gpu_time"
    assert v["outcome"] == "regressed"


def test_verdict_neutral_when_kernel_time_within_noise_band() -> None:
    base = _csv(
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker,gpu__time_duration.sum,usecond,1000.0",
        "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,40.0",
    )
    opt = _csv(
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker,gpu__time_duration.sum,usecond,990.0",  # ~1% — inside the noise band
        "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,40.0",
    )
    v = at.diff_ncu_runs(base, opt)["verdict"]
    assert v["basis"] == "kernel_gpu_time"
    assert v["outcome"] == "neutral"


def test_kernel_time_unavailable_falls_back_to_bottleneck_basis() -> None:
    # The legacy fixtures carry no duration metric → verdict basis is the bottleneck diff.
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, OPTIMIZED_MEMORY_RESOLVED)
    assert result["kernel_time"]["available"] is False
    assert result["verdict"]["basis"] == "bottleneck_diff"
    # And the legacy outcome is preserved.
    assert result["verdict"]["outcome"] == "improved"


def test_verdict_surfaces_disagreement_between_kernel_time_and_bottlenecks() -> None:
    # Bottleneck resolves (uncoalesced gone) but kernel time barely moves: the headline
    # outcome follows kernel time, while bottleneck_outcome records the discrepancy.
    base = _csv(
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker,gpu__time_duration.sum,usecond,1000.0",
        "ker,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld,sector,8.0",
    )
    opt = _csv(
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker,gpu__time_duration.sum,usecond,995.0",  # essentially unchanged
        "ker,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld,sector,1.0",
    )
    v = at.diff_ncu_runs(base, opt)["verdict"]
    assert v["outcome"] == "neutral"            # from kernel time
    assert v["bottleneck_outcome"] == "improved"  # uncoalesced_access resolved


# ── Embedded full analyses ────────────────────────────────────────────────────

def test_embedded_analyses_are_ncu_analysis_v1() -> None:
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, OPTIMIZED_MEMORY_RESOLVED)
    assert result["baseline"]["schema"] == "ncu_analysis_v1"
    assert result["optimized"]["schema"] == "ncu_analysis_v1"


def test_embedded_analyses_have_recommendations() -> None:
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, OPTIMIZED_MEMORY_RESOLVED)
    assert isinstance(result["baseline"]["recommendations"], list)
    assert isinstance(result["optimized"]["recommendations"], list)


# ── Metric delta label and unit fields ────────────────────────────────────────

def test_metric_deltas_have_label_field() -> None:
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, OPTIMIZED_MEMORY_RESOLVED)
    deltas = result["metric_deltas"]
    for key, delta in deltas.items():
        assert "label" in delta, f"metric_deltas[{key!r}] missing 'label'"
        assert isinstance(delta["label"], str) and delta["label"], (
            f"metric_deltas[{key!r}]['label'] must be a non-empty string"
        )


def test_metric_deltas_have_unit_field() -> None:
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, OPTIMIZED_MEMORY_RESOLVED)
    deltas = result["metric_deltas"]
    for key, delta in deltas.items():
        assert "unit" in delta, f"metric_deltas[{key!r}] missing 'unit'"
        assert isinstance(delta["unit"], str), (
            f"metric_deltas[{key!r}]['unit'] must be a string"
        )


def test_metric_deltas_dram_has_pct_unit() -> None:
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, OPTIMIZED_MEMORY_RESOLVED)
    d = result["metric_deltas"]["avg_dram_throughput_pct"]
    assert d["unit"] == "%"
    assert d["label"] == "DRAM Throughput"


def test_metric_deltas_stall_fraction_has_empty_unit() -> None:
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, OPTIMIZED_MEMORY_RESOLVED)
    d = result["metric_deltas"]["memory_stall_fraction"]
    assert d["unit"] == ""
    assert d["label"] == "Memory Stall Fraction"


def test_metric_deltas_load_sectors_per_request_present() -> None:
    # avg_global_load_sectors_per_request was added alongside label/unit
    result = at.diff_ncu_runs(BASELINE_MEMORY_BOUND, OPTIMIZED_MEMORY_RESOLVED)
    # The metric may be absent if neither CSV has the raw sectors metric,
    # but the key must be recognised and not raise
    deltas = result["metric_deltas"]
    if "avg_global_load_sectors_per_request" in deltas:
        d = deltas["avg_global_load_sectors_per_request"]
        assert d["unit"] == ""
        assert d["label"] == "Load Sectors / Request"
