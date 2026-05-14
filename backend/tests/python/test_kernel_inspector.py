import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as at


def test_occupancy_estimation_uses_register_and_shared_memory_limits() -> None:
    estimate = at.estimate_occupancy(
        registers_per_thread=64,
        shared_memory_per_block_bytes=32768,
        threads_per_block=256,
        device_limits={
            "warp_size": 32,
            "max_threads_per_sm": 2048,
            "max_blocks_per_sm": 32,
            "registers_per_sm": 65536,
            "shared_memory_per_sm_bytes": 98304,
        },
    )

    assert estimate["occupancy_pct"] == 37.5
    assert estimate["active_blocks_per_sm"] == 3
    assert estimate["active_warps_per_sm"] == 24
    assert estimate["limiting_factors"] == ["shared_memory"]


def test_pytorch_profiler_kernel_attrs_surface_launch_resources() -> None:
    trace = at.PytorchProfilerTrace.from_json_payload(
        {
            "traceEvents": [
                {
                    "name": "void fused_elementwise(float*)",
                    "cat": "cuda_kernel",
                    "ph": "X",
                    "ts": 10,
                    "dur": 5,
                    "args": {
                        "device": 0,
                        "registers_per_thread": 32,
                        "shared_memory_per_block_bytes": 1024,
                        "blockDim.x": 128,
                        "blockDim.y": 1,
                        "blockDim.z": 1,
                        "gridDim.x": 80,
                        "gridDim.y": 1,
                        "gridDim.z": 1,
                    },
                }
            ]
        }
    )

    events, _ = at.map_pytorch_profiler_to_ir(trace, run_id="run_kernel_attrs")
    summary = at.launch_summary_from_attrs(events[0].attrs)

    assert events[0].attrs["registers_per_thread"] == 32
    assert summary.threads_per_block == 128
    assert summary.grid_dims == (80, 1, 1)
    assert summary.occupancy_estimate["occupancy_pct"] is not None


def test_nsight_compute_csv_import_maps_kernel_metrics_to_ir(tmp_path: Path) -> None:
    csv_path = tmp_path / "ncu.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Kernel Name,Metric Name,Metric Unit,Metric Value",
                "ampere_sgemm,launch__registers_per_thread,register/thread,64",
                "ampere_sgemm,launch__shared_mem_per_block_static,byte/block,16384",
                "ampere_sgemm,launch__shared_mem_per_block_dynamic,byte/block,16384",
                "ampere_sgemm,launch__block_size,thread/block,256",
                "ampere_sgemm,launch__grid_dim_x,,120",
            ]
        ),
        encoding="utf-8",
    )

    summaries = at.parse_nsight_compute_csv(csv_path)
    events, metrics = at.map_nsight_compute_csv_to_ir(csv_path, run_id="run_ncu")

    assert len(summaries) == 1
    assert summaries[0].registers_per_thread == 64
    assert summaries[0].shared_memory_per_block_bytes == 32768
    assert summaries[0].threads_per_block == 256
    assert summaries[0].grid_dims == (120, 1, 1)
    assert events[0].attrs["occupancy_estimate"]["occupancy_pct"] == 37.5
    assert any(metric.metric_name == "kernel.registers_per_thread" for metric in metrics)


def test_ncu_metric_aliases_used_by_real_cuda_integration() -> None:
    text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "global_heavy,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,81.0",
        "global_heavy,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,%,12.0",
        "global_heavy,l1tex__t_sector_hit_rate.pct,%,33.0",
        "global_heavy,lts__t_sector_hit_rate.pct,%,48.0",
        "global_heavy,smsp__issue_active.avg.pct_of_peak_sustained_active,%,57.0",
        "global_heavy,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,%,31.0",
    ])

    summary = at.parse_nsight_compute_csv_text(text)[0]

    assert summary.dram_throughput_pct == 81.0
    assert summary.tensor_core_utilization_pct == 12.0
    assert summary.l1_cache_hit_rate_pct == 33.0
    assert summary.l2_cache_hit_rate_pct == 48.0
    assert summary.issue_slot_utilization_pct == 57.0
    assert summary.warp_stall_breakdown["long_scoreboard"] == 31.0
    assert summary.dominant_warp_stall == "long_scoreboard"


def test_common_ir_run_summary_includes_kernel_launch_summary() -> None:
    event = at.EventRecord(
        event_id="kernel_1",
        run_id="run_kernel_summary",
        event_family="kernel",
        event_type="cuda_kernel",
        ts_start_ns=0,
        ts_end_ns=10_000,
        duration_ns=10_000,
        source="unit",
        device_id="gpu0",
        step_id="step_1",
        attrs={
            "kernel_name_raw": "tiny_kernel",
            "registers_per_thread": 16,
            "shared_memory_per_block_bytes": 0,
            "threads_per_block": 128,
        },
    )
    run = at.RunRecord(
        run_id="run_kernel_summary",
        job=at.JobInfo(job_id="job_1", workload_class="training", status="completed"),
        workload=at.WorkloadInfo(),
        events=[event],
    )

    summary = at.summarize_ir_run(run)

    kernel_summary = summary["run_summary"]["kernel_launch_summary"]
    assert kernel_summary["kernel_count"] == 1
    assert kernel_summary["kernels_with_launch_metadata"] == 1
    assert kernel_summary["kernels"][0]["registers_per_thread"] == 16
