from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NcuMetricPreset:
    name: str
    description: str
    metrics: tuple[str, ...]
    required_canonical_metrics: tuple[str, ...]


NCU_METRIC_PRESETS: dict[str, NcuMetricPreset] = {
    "memory": NcuMetricPreset(
        name="memory",
        description="DRAM bandwidth, cache hit rates, coalescing efficiency, and memory-related stalls.",
        metrics=(
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "l1tex__t_sector_hit_rate.pct",
            "lts__t_sector_hit_rate.pct",
            "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld",
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle",
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_long_scoreboard",
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_mio",
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_lg",
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_texture",
        ),
        required_canonical_metrics=(
            "dram_throughput_pct",
            "l1_cache_hit_rate_pct",
            "l2_cache_hit_rate_pct",
            "warp_stall_memory_throttle",
            "warp_stall_long_scoreboard",
        ),
    ),
    "tensor": NcuMetricPreset(
        name="tensor",
        description="Tensor core utilization and basic achieved occupancy context.",
        metrics=(
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "sm__issue_active.avg.pct_of_peak_sustained_active",
        ),
        required_canonical_metrics=(
            "tensor_core_utilization_pct",
            "achieved_occupancy_pct",
            "issue_slot_utilization_pct",
        ),
    ),
    "occupancy": NcuMetricPreset(
        name="occupancy",
        description="Achieved occupancy plus launch resources needed to explain occupancy limits.",
        metrics=(
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "launch__block_size",
            "launch__registers_per_thread",
            "launch__shared_mem_per_block_static",
            "launch__shared_mem_per_block_dynamic",
        ),
        required_canonical_metrics=(
            "achieved_occupancy_pct",
            "threads_per_block",
            "registers_per_thread",
        ),
    ),
    "stalls": NcuMetricPreset(
        name="stalls",
        description="Warp stall reasons for memory, synchronization, and scheduling pressure.",
        metrics=(
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle",
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_long_scoreboard",
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_mio",
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_lg",
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_texture",
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_barrier",
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_wait",
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_short_scoreboard",
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_dispatch",
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_not_selected",
            "smsp__warps_eligible.avg.per_cycle_active",
            "smsp__warps_active.avg.pct_of_peak_sustained_active",
            "sm__issue_active.avg.pct_of_peak_sustained_active",
        ),
        required_canonical_metrics=(
            "warp_stall_memory_throttle",
            "warp_stall_long_scoreboard",
            "warp_stall_barrier",
            "warp_stall_wait",
            "eligible_warps_per_scheduler",
            "scheduler_active_pct",
            "issue_slot_utilization_pct",
        ),
    ),
}


NCU_METRIC_PRESETS["full"] = NcuMetricPreset(
    name="full",
    description="Union of memory, tensor, occupancy, and stall presets.",
    metrics=tuple(dict.fromkeys(
        metric
        for preset_name in ("memory", "tensor", "occupancy", "stalls")
        for metric in NCU_METRIC_PRESETS[preset_name].metrics
    )),
    required_canonical_metrics=tuple(dict.fromkeys(
        metric
        for preset_name in ("memory", "tensor", "occupancy", "stalls")
        for metric in NCU_METRIC_PRESETS[preset_name].required_canonical_metrics
    )),
)


def get_ncu_preset(name: str) -> NcuMetricPreset:
    key = name.strip().lower()
    try:
        return NCU_METRIC_PRESETS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(NCU_METRIC_PRESETS))
        raise ValueError(f"unknown NCU preset {name!r}; choose one of: {choices}") from exc


def build_ncu_command(
    preset_name: str,
    workload_command: list[str],
    *,
    output: str | None = None,
    kernel_name: str | None = None,
    target_processes: str = "all",
    launch_skip: int | None = None,
    launch_count: int | None = None,
) -> list[str]:
    preset = get_ncu_preset(preset_name)
    command = [
        "ncu",
        "--csv",
        "--target-processes",
        target_processes,
        "--metrics",
        ",".join(preset.metrics),
    ]
    if kernel_name:
        command.extend(["--kernel-name", kernel_name])
    if launch_skip is not None:
        command.extend(["--launch-skip", str(launch_skip)])
    if launch_count is not None:
        command.extend(["--launch-count", str(launch_count)])
    command.extend(workload_command)
    if output:
        command.extend([">", output])
    return command


def format_shell_command(command: list[str]) -> str:
    return " ".join(_quote_part(part) for part in command)


def describe_ncu_presets() -> list[dict[str, Any]]:
    return [
        {
            "name": preset.name,
            "description": preset.description,
            "metrics": list(preset.metrics),
            "required_canonical_metrics": list(preset.required_canonical_metrics),
        }
        for preset in NCU_METRIC_PRESETS.values()
    ]


def _quote_part(part: str) -> str:
    if part == ">":
        return part
    if not part or any(ch.isspace() for ch in part) or any(ch in part for ch in "\"'&|()<>"):
        escaped = part.replace('"', '\\"')
        return f'"{escaped}"'
    return part
