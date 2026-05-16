import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as at


# ── Helpers ───────────────────────────────────────────────────────────────────

def _csv(*rows: str) -> str:
    return "\n".join(rows)


def _long_format_csv(*metric_rows: tuple[str, str, str, str]) -> str:
    lines = ["Kernel Name,Metric Name,Metric Unit,Metric Value"]
    for kernel, metric, unit, value in metric_rows:
        lines.append(f"{kernel},{metric},{unit},{value}")
    return "\n".join(lines)


LAUNCH_CSV = _csv(
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "ampere_sgemm,launch__registers_per_thread,register/thread,64",
    "ampere_sgemm,launch__shared_mem_per_block_static,byte/block,16384",
    "ampere_sgemm,launch__block_size,thread/block,256",
    "ampere_sgemm,launch__grid_dim_x,,120",
)


# ── Metric parsing ────────────────────────────────────────────────────────────

def test_parse_warp_stall_metrics() -> None:
    text = _long_format_csv(
        ("my_kernel", "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle", "%", "35.5"),
        ("my_kernel", "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_long_scoreboard", "%", "20.1"),
        ("my_kernel", "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_barrier", "%", "5.0"),
    )
    summaries = at.parse_nsight_compute_csv_text(text)

    assert len(summaries) == 1
    s = summaries[0]
    assert s.dominant_warp_stall == "memory_throttle"
    assert abs(s.dominant_warp_stall_pct - 35.5) < 0.01
    assert "memory_throttle" in s.warp_stall_breakdown
    assert "long_scoreboard" in s.warp_stall_breakdown
    assert "barrier" in s.warp_stall_breakdown


def test_parse_warp_stall_metrics_text() -> None:
    text = _long_format_csv(
        ("ker", "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle", "%", "42.0"),
        ("ker", "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_barrier", "%", "8.0"),
    )
    summaries = at.parse_nsight_compute_csv_text(text)
    assert summaries[0].dominant_warp_stall == "memory_throttle"
    assert summaries[0].warp_stall_breakdown.get("memory_throttle") == 42.0


def test_parse_dram_throughput() -> None:
    text = _long_format_csv(
        ("ker", "dram__throughput.avg.pct_of_peak_sustained_elapsed", "%", "78.5"),
    )
    summaries = at.parse_nsight_compute_csv_text(text)
    assert summaries[0].dram_throughput_pct == 78.5


def test_parse_tensor_core_utilization() -> None:
    text = _long_format_csv(
        ("ker", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "%", "62.3"),
    )
    summaries = at.parse_nsight_compute_csv_text(text)
    assert summaries[0].tensor_core_utilization_pct == 62.3


def test_parse_cache_hit_rates() -> None:
    text = _long_format_csv(
        ("ker", "l1tex__t_sector_hit_rate.pct", "%", "55.0"),
        ("ker", "lts__t_sector_hit_rate.pct", "%", "70.0"),
    )
    summaries = at.parse_nsight_compute_csv_text(text)
    assert summaries[0].l1_cache_hit_rate_pct == 55.0
    assert summaries[0].l2_cache_hit_rate_pct == 70.0


def test_parse_issue_slot_utilization() -> None:
    text = _long_format_csv(
        ("ker", "sm__issue_active.avg.pct_of_peak_sustained_active", "%", "38.0"),
    )
    summaries = at.parse_nsight_compute_csv_text(text)
    assert summaries[0].issue_slot_utilization_pct == 38.0


def test_parse_scheduler_metrics() -> None:
    text = _long_format_csv(
        ("ker", "smsp__warps_eligible.avg.per_cycle_active", "warp", "0.5"),
        ("ker", "smsp__warps_active.avg.pct_of_peak_sustained_active", "%", "31.0"),
    )
    summaries = at.parse_nsight_compute_csv_text(text)
    assert summaries[0].eligible_warps_per_scheduler == 0.5
    assert summaries[0].scheduler_active_pct == 31.0


def test_parse_memory_throughput_column_name() -> None:
    # Alternative column spelling used by some NCU versions
    text = _long_format_csv(
        ("ker", "Memory Throughput", "%", "80.0"),
    )
    summaries = at.parse_nsight_compute_csv_text(text)
    assert summaries[0].dram_throughput_pct == 80.0


# ── derive_ncu_run_summary ────────────────────────────────────────────────────

def test_derive_ncu_run_summary_empty() -> None:
    result = at.derive_ncu_run_summary([])
    assert result["kernel_count"] == 0
    assert result["kernels_with_ncu_data"] == 0
    assert result["dominant_warp_stall"] == "unknown"


def test_derive_ncu_run_summary_averages() -> None:
    text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker_a,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,80.0",
        "ker_a,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,%,20.0",
        "ker_a,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,40.0",
        "ker_b,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,60.0",
        "ker_b,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,%,30.0",
        "ker_b,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,25.0",
    ])
    summaries = at.parse_nsight_compute_csv_text(text)
    result = at.derive_ncu_run_summary(summaries)

    assert result["kernel_count"] == 2
    assert result["avg_dram_throughput_pct"] == 70.0
    assert result["avg_tensor_core_utilization_pct"] == 25.0
    assert result["dominant_warp_stall"] == "memory_throttle"
    # Both kernels have only memory_throttle stalls (40% and 25%).
    # New semantics: avg(40, 25) / 100 = 0.325
    assert result["memory_stall_fraction"] == 0.325
    assert result["kernels_with_ncu_data"] == 2


def test_derive_ncu_run_summary_sync_stall_fraction() -> None:
    text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker_a,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_barrier,%,35.0",
        "ker_b,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,40.0",
    ])
    summaries = at.parse_nsight_compute_csv_text(text)
    result = at.derive_ncu_run_summary(summaries)

    # ker_b has 40% memory stall; ker_a has 0% (barrier is sync, not memory).
    # New semantics: avg(0, 40) / 100 = 0.20
    assert result["memory_stall_fraction"] == 0.2
    assert result["compute_stall_fraction"] == 0.0


def test_derive_ncu_run_summary_multi_kernel_partial_metrics() -> None:
    text = _long_format_csv(
        ("mem_a", "dram__throughput.avg.pct_of_peak_sustained_elapsed", "%", "90.0"),
        ("mem_a", "l1tex__t_sector_hit_rate.pct", "%", "24.0"),
        ("mem_a", "lts__t_sector_hit_rate.pct", "%", "44.0"),
        ("mem_a", "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle", "%", "45.0"),
        ("mem_a", "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_long_scoreboard", "%", "15.0"),
        ("mem_b", "dram__throughput.avg.pct_of_peak_sustained_elapsed", "%", "78.0"),
        ("mem_b", "l1tex__t_sector_hit_rate.pct", "%", "36.0"),
        ("mem_b", "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle", "%", "35.0"),
        ("tensor_only", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "%", "18.0"),
        ("tensor_only", "sm__issue_active.avg.pct_of_peak_sustained_active", "%", "72.0"),
        ("l2_only", "lts__t_sector_hit_rate.pct", "%", "42.0"),
        ("sync_only", "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_barrier", "%", "30.0"),
        ("sync_only", "sm__issue_active.avg.pct_of_peak_sustained_active", "%", "55.0"),
        ("launch_only", "launch__registers_per_thread", "register/thread", "64"),
        ("launch_only", "launch__block_size", "thread/block", "256"),
    )
    summaries = at.parse_nsight_compute_csv_text(text)
    result = at.derive_ncu_run_summary(summaries)

    assert result["kernel_count"] == 6
    assert result["kernels_with_ncu_data"] == 5
    assert result["kernels_with_warp_stall_data"] == 3
    assert result["avg_dram_throughput_pct"] == 84.0
    assert result["avg_tensor_core_utilization_pct"] == 18.0
    assert result["avg_l1_cache_hit_rate_pct"] == 30.0
    assert result["avg_l2_cache_hit_rate_pct"] == 43.0
    assert result["avg_issue_slot_utilization_pct"] == 63.5
    assert result["dominant_warp_stall"] == "memory_throttle"
    assert result["dominant_warp_stall_pct"] == 26.67
    assert result["warp_stall_breakdown"] == {
        "barrier": 10.0,
        "long_scoreboard": 5.0,
        "memory_throttle": 26.67,
    }
    assert result["memory_stall_fraction"] == 0.3167
    assert result["compute_stall_fraction"] == 0.0


# ── classify_ncu_bottlenecks ──────────────────────────────────────────────────

def test_classify_memory_bandwidth_bound() -> None:
    # tc=60% is healthy so memory_bandwidth_bound scores highest (0.95 vs 0.40 for isu)
    summary = {
        "kernel_count": 5,
        "kernels_with_ncu_data": 5,
        "avg_dram_throughput_pct": 95.0,
        "avg_tensor_core_utilization_pct": 60.0,
        "avg_l1_cache_hit_rate_pct": 35.0,
        "avg_l2_cache_hit_rate_pct": 45.0,
        "avg_issue_slot_utilization_pct": 65.0,
        "avg_occupancy_pct": 75.0,
        "dominant_warp_stall": "memory_throttle",
        "dominant_warp_stall_pct": 42.0,
        "warp_stall_breakdown": {"memory_throttle": 42.0},
        "memory_stall_fraction": 0.8,
        "compute_stall_fraction": 0.0,
    }
    bottlenecks = at.classify_ncu_bottlenecks(summary)
    labels = [b["label"] for b in bottlenecks]
    assert bottlenecks[0]["label"] == "memory_bandwidth_bound"
    assert "l1_cache_thrashing" in labels
    assert "l2_cache_thrashing" in labels
    assert "warp_stall_memory" in labels


def test_classify_tensor_core_underutilized() -> None:
    summary = {
        "kernel_count": 3,
        "kernels_with_ncu_data": 3,
        "avg_dram_throughput_pct": 40.0,
        "avg_tensor_core_utilization_pct": 8.0,
        "avg_l1_cache_hit_rate_pct": 70.0,
        "avg_l2_cache_hit_rate_pct": 80.0,
        "avg_issue_slot_utilization_pct": 70.0,
        "avg_occupancy_pct": 65.0,
        "dominant_warp_stall": "not_selected",
        "dominant_warp_stall_pct": 12.0,
        "warp_stall_breakdown": {},
        "memory_stall_fraction": 0.0,
        "compute_stall_fraction": 1.0,
    }
    bottlenecks = at.classify_ncu_bottlenecks(summary)
    labels = [b["label"] for b in bottlenecks]
    assert "tensor_core_underutilized" in labels


def test_classify_warp_stall_sync() -> None:
    summary = {
        "kernel_count": 2,
        "kernels_with_ncu_data": 2,
        "avg_dram_throughput_pct": 30.0,
        "avg_tensor_core_utilization_pct": 50.0,
        "avg_l1_cache_hit_rate_pct": 75.0,
        "avg_l2_cache_hit_rate_pct": 85.0,
        "avg_issue_slot_utilization_pct": 65.0,
        "avg_occupancy_pct": 70.0,
        "dominant_warp_stall": "barrier",
        "dominant_warp_stall_pct": 28.0,
        "warp_stall_breakdown": {"barrier": 28.0},
        "memory_stall_fraction": 0.0,
        "compute_stall_fraction": 0.0,
    }
    bottlenecks = at.classify_ncu_bottlenecks(summary)
    labels = [b["label"] for b in bottlenecks]
    assert "warp_stall_sync" in labels


def test_classify_low_issue_efficiency() -> None:
    summary = {
        "kernel_count": 2,
        "kernels_with_ncu_data": 2,
        "avg_dram_throughput_pct": 25.0,
        "avg_tensor_core_utilization_pct": 60.0,
        "avg_l1_cache_hit_rate_pct": 80.0,
        "avg_l2_cache_hit_rate_pct": 90.0,
        "avg_issue_slot_utilization_pct": 35.0,
        "avg_occupancy_pct": 50.0,
        "dominant_warp_stall": "short_scoreboard",
        "dominant_warp_stall_pct": 18.0,
        "warp_stall_breakdown": {},
        "memory_stall_fraction": 0.0,
        "compute_stall_fraction": 1.0,
    }
    bottlenecks = at.classify_ncu_bottlenecks(summary)
    labels = [b["label"] for b in bottlenecks]
    assert "low_issue_efficiency" in labels


def test_classify_occupancy_limited_by_registers() -> None:
    text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker,sm__warps_active.avg.pct_of_peak_sustained_active,%,25.0",
        "ker,launch__block_size,thread/block,256",
        "ker,launch__registers_per_thread,register/thread,128",
    ])
    result = at.analyze_ncu_csv_text(text)
    labels = [b["label"] for b in result["bottlenecks"]]
    assert "occupancy_limited" in labels
    assert "occupancy_limited_by_registers" in labels
    assert "registers" in result["ncu_run_summary"]["occupancy_limit_causes"]


def test_classify_occupancy_limited_by_shared_memory() -> None:
    text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker,sm__warps_active.avg.pct_of_peak_sustained_active,%,25.0",
        "ker,launch__block_size,thread/block,128",
        "ker,launch__shared_mem_per_block_static,byte/block,65536",
    ])
    result = at.analyze_ncu_csv_text(text)
    labels = [b["label"] for b in result["bottlenecks"]]
    assert "occupancy_limited" in labels
    assert "occupancy_limited_by_shared_memory" in labels
    assert "shared_memory" in result["ncu_run_summary"]["occupancy_limit_causes"]


def test_classify_low_warp_scheduler_utilization() -> None:
    text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker,smsp__warps_eligible.avg.per_cycle_active,warp,0.4",
        "ker,smsp__warps_active.avg.pct_of_peak_sustained_active,%,35.0",
        "ker,sm__issue_active.avg.pct_of_peak_sustained_active,%,45.0",
    ])
    result = at.analyze_ncu_csv_text(text)
    labels = [b["label"] for b in result["bottlenecks"]]
    assert "low_warp_scheduler_utilization" in labels
    assert result["ncu_run_summary"]["avg_eligible_warps_per_scheduler"] == 0.4


def test_classify_l1_cache_thrashing_fires_independently() -> None:
    summary = {
        "kernel_count": 2,
        "kernels_with_ncu_data": 2,
        "avg_dram_throughput_pct": 30.0,
        "avg_tensor_core_utilization_pct": 50.0,
        "avg_l1_cache_hit_rate_pct": 25.0,  # below 40 → l1_cache_thrashing
        "avg_l2_cache_hit_rate_pct": 65.0,  # above 50 → no l2_cache_thrashing
        "avg_global_load_sectors_per_request": None,
        "avg_issue_slot_utilization_pct": 70.0,
        "avg_occupancy_pct": 75.0,
        "dominant_warp_stall": "not_selected",
        "dominant_warp_stall_pct": 5.0,
        "warp_stall_breakdown": {},
        "memory_stall_fraction": 0.0,
        "compute_stall_fraction": 0.0,
    }
    labels = [b["label"] for b in at.classify_ncu_bottlenecks(summary)]
    assert "l1_cache_thrashing" in labels
    assert "l2_cache_thrashing" not in labels


def test_classify_l2_cache_thrashing_fires_independently() -> None:
    summary = {
        "kernel_count": 2,
        "kernels_with_ncu_data": 2,
        "avg_dram_throughput_pct": 30.0,
        "avg_tensor_core_utilization_pct": 50.0,
        "avg_l1_cache_hit_rate_pct": 55.0,  # above 40 → no l1_cache_thrashing
        "avg_l2_cache_hit_rate_pct": 35.0,  # below 50 → l2_cache_thrashing
        "avg_global_load_sectors_per_request": None,
        "avg_issue_slot_utilization_pct": 70.0,
        "avg_occupancy_pct": 75.0,
        "dominant_warp_stall": "not_selected",
        "dominant_warp_stall_pct": 5.0,
        "warp_stall_breakdown": {},
        "memory_stall_fraction": 0.0,
        "compute_stall_fraction": 0.0,
    }
    labels = [b["label"] for b in at.classify_ncu_bottlenecks(summary)]
    assert "l2_cache_thrashing" in labels
    assert "l1_cache_thrashing" not in labels


def test_classify_uncoalesced_access_from_sectors_per_request() -> None:
    text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld,sector/request,12.0",
    ])
    result = at.analyze_ncu_csv_text(text)
    labels = [b["label"] for b in result["bottlenecks"]]
    assert "uncoalesced_access" in labels
    summary = result["ncu_run_summary"]
    assert summary["avg_global_load_sectors_per_request"] == 12.0


def test_classify_uncoalesced_access_not_fired_when_coalesced() -> None:
    summary = {
        "kernel_count": 1,
        "kernels_with_ncu_data": 1,
        "avg_dram_throughput_pct": 40.0,
        "avg_tensor_core_utilization_pct": 50.0,
        "avg_l1_cache_hit_rate_pct": 70.0,
        "avg_l2_cache_hit_rate_pct": 80.0,
        "avg_global_load_sectors_per_request": 2.0,  # ≤4 → no bottleneck
        "avg_issue_slot_utilization_pct": 70.0,
        "avg_occupancy_pct": 75.0,
        "dominant_warp_stall": "not_selected",
        "dominant_warp_stall_pct": 5.0,
        "warp_stall_breakdown": {},
        "memory_stall_fraction": 0.0,
        "compute_stall_fraction": 0.0,
    }
    labels = [b["label"] for b in at.classify_ncu_bottlenecks(summary)]
    assert "uncoalesced_access" not in labels


def test_classify_insufficient_ncu_data() -> None:
    summary = at.derive_ncu_run_summary([])
    bottlenecks = at.classify_ncu_bottlenecks(summary)
    assert len(bottlenecks) == 1
    assert bottlenecks[0]["label"] == "insufficient_ncu_data"
    assert bottlenecks[0]["score"] == 1.0


# ── Full pipeline ─────────────────────────────────────────────────────────────

def test_analyze_ncu_csv_text_returns_recommendations() -> None:
    text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "big_gemm,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,85.0",
        "big_gemm,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,%,5.0",
        "big_gemm,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,42.0",
        "big_gemm,l1tex__t_sector_hit_rate.pct,%,30.0",
        "big_gemm,lts__t_sector_hit_rate.pct,%,40.0",
    ])
    result = at.analyze_ncu_csv_text(text, environment={"mixed_precision": False})

    assert result["schema"] == "ncu_analysis_v1"
    assert result["kernel_count"] == 1
    assert result["diagnostic_scope"]["type"] == "measured_ncu"
    assert result["primary_bottleneck"] is not None
    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) > 0
    rec_ids = {r["id"] for r in result["recommendations"]}
    assert len(rec_ids) > 0


def test_analyze_ncu_csv_text_empty_returns_insufficient_data() -> None:
    result = at.analyze_ncu_csv_text("")
    assert result["primary_bottleneck"] == "insufficient_ncu_data"
    assert result["validation"]["valid"] is False
    assert result["validation"]["errors"]


def test_validate_ncu_csv_reports_missing_preset_metrics() -> None:
    text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,88.0",
    ])
    result = at.analyze_ncu_csv_text(text)

    validation = result["validation"]
    assert validation["valid"] is True
    assert "dram_throughput_pct" in validation["present_metrics"]
    assert "l1_cache_hit_rate_pct" in validation["missing_metrics_by_preset"]["memory"]
    assert any("Memory diagnosis may be incomplete" in warning for warning in validation["warnings"])


def test_validate_ncu_csv_reports_malformed_shape() -> None:
    validation = at.validate_ncu_csv_text("not,ncu\n1,2")

    assert validation["valid"] is False
    assert "CSV is missing a kernel name column." in validation["errors"]
    assert "CSV is missing a metric name column." in validation["errors"]
    assert "CSV is missing a metric value column." in validation["errors"]


def test_ncu_presets_build_command_and_describe_metrics() -> None:
    preset = at.get_ncu_preset("memory")
    command = at.build_ncu_command("memory", ["./app"], output="ncu_memory.csv", kernel_name="regex:my_kernel")
    shell = at.format_shell_command(command)

    assert "dram__throughput.avg.pct_of_peak_sustained_elapsed" in preset.metrics
    assert "--metrics" in command
    assert "--kernel-name regex:my_kernel" in shell
    assert "> ncu_memory.csv" in shell


def test_achieved_occupancy_metric_is_used_in_summary() -> None:
    text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker,sm__warps_active.avg.pct_of_peak_sustained_active,%,37.5",
        "ker,launch__block_size,thread/block,256",
    ])
    result = at.analyze_ncu_csv_text(text)

    assert result["ncu_run_summary"]["avg_occupancy_pct"] == 37.5


def test_analyze_ncu_existing_launch_fields_preserved() -> None:
    summaries = at.parse_nsight_compute_csv_text(LAUNCH_CSV)

    # Existing launch fields should still work after adding new NCU fields
    assert summaries[0].registers_per_thread == 64
    assert summaries[0].shared_memory_per_block_bytes == 16384
    assert summaries[0].threads_per_block == 256


# ── Gap fixes ─────────────────────────────────────────────────────────────────

def test_memory_stall_fraction_uses_magnitude_not_dominance() -> None:
    """memory_stall_fraction must reflect actual stall percentages, not just whether
    memory is the dominant stall type.  5% memory stalls must not produce fraction=1.0."""
    text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker_a,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,5.0",
        "ker_b,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,5.0",
    ])
    summaries = at.parse_nsight_compute_csv_text(text)
    result = at.derive_ncu_run_summary(summaries)

    # avg(5, 5) / 100 = 0.05 — both kernels have memory_throttle as dominant at only 5%
    assert result["memory_stall_fraction"] == 0.05, (
        f"expected 0.05 for 5% stalls, got {result['memory_stall_fraction']}"
    )


def test_parse_realistic_ncu_csv_prof_prefix_lines() -> None:
    """==PROF== metadata lines emitted by NCU must be stripped before CSV parsing."""
    text = "\n".join([
        "==PROF== Connected to process 12345 (/usr/bin/myapp)",
        "==PROF== Profiling \"my_kernel\" - 0: 0%....50%....100% - 8 passes",
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "my_kernel,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,82.0",
        "my_kernel,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,%,15.0",
        "my_kernel,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,38.0",
    ])
    summaries = at.parse_nsight_compute_csv_text(text)

    assert len(summaries) == 1
    s = summaries[0]
    assert s.kernel_name == "my_kernel"
    assert s.dram_throughput_pct == 82.0
    assert s.tensor_core_utilization_pct == 15.0
    assert s.dominant_warp_stall == "memory_throttle"


def test_parse_realistic_ncu_csv_extra_columns() -> None:
    """Real NCU exports include ID / Process ID / Section Name columns; parser must ignore them."""
    text = "\n".join([
        "ID,Process ID,Process Name,Host Name,Kernel Name,Kernel Time,Context,Stream,Section Name,Metric Name,Metric Unit,Metric Value",
        "0,12345,/usr/bin/app,hostname,my_gemm,2024-01-01,1,7,Memory Workload,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,75.0",
        "0,12345,/usr/bin/app,hostname,my_gemm,2024-01-01,1,7,Warp State Stats,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,38.0",
        "0,12345,/usr/bin/app,hostname,my_gemm,2024-01-01,1,7,Warp State Stats,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_long_scoreboard,%,22.0",
    ])
    summaries = at.parse_nsight_compute_csv_text(text)

    assert len(summaries) == 1
    s = summaries[0]
    assert s.dram_throughput_pct == 75.0
    assert s.dominant_warp_stall == "memory_throttle"
    assert abs(s.dominant_warp_stall_pct - 38.0) < 0.01
    assert "long_scoreboard" in s.warp_stall_breakdown


def test_tensor_core_underutilized_end_to_end() -> None:
    """Full CSV → classify pipeline must detect tensor_core_underutilized when tc < 30%."""
    text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        # launch config → occupancy > 40%
        "tc_kernel,launch__block_size,thread/block,256",
        "tc_kernel,launch__registers_per_thread,register/thread,32",
        # tensor core utilization below threshold
        "tc_kernel,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,%,12.0",
        "tc_kernel,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,45.0",
        "tc_kernel,l1tex__t_sector_hit_rate.pct,%,72.0",
        "tc_kernel,lts__t_sector_hit_rate.pct,%,80.0",
    ])
    result = at.analyze_ncu_csv_text(text)

    labels = [b["label"] for b in result["bottlenecks"]]
    assert "tensor_core_underutilized" in labels, f"expected tensor_core_underutilized in {labels}"
    tc_b = next(b for b in result["bottlenecks"] if b["label"] == "tensor_core_underutilized")
    assert tc_b["evidence"]["avg_tensor_core_utilization_pct"] == 12.0


def test_parse_ncu_csv_combined_prof_prefix_and_extra_columns() -> None:
    """Combined real-NCU scenario: ==PROF== lines + extra ID/Process ID columns."""
    text = "\n".join([
        "==PROF== Connected to process 99999 (/bin/app)",
        "==PROF== Profiling \"big_kernel\" - 0: 0%....100% - 4 passes",
        "ID,Process ID,Kernel Name,Metric Name,Metric Unit,Metric Value",
        "0,99999,big_kernel,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,88.0",
        "0,99999,big_kernel,l1tex__t_sector_hit_rate.pct,%,28.0",
        "0,99999,big_kernel,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,45.0",
    ])
    summaries = at.parse_nsight_compute_csv_text(text)

    assert len(summaries) == 1
    s = summaries[0]
    assert s.dram_throughput_pct == 88.0
    assert s.l1_cache_hit_rate_pct == 28.0
    assert s.dominant_warp_stall == "memory_throttle"
