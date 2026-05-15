from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .common_ir import EventRecord, MetricRecord


DEFAULT_DEVICE_LIMITS = {
    "warp_size": 32,
    "max_threads_per_sm": 2048,
    "max_blocks_per_sm": 32,
    "registers_per_sm": 65536,
    "shared_memory_per_sm_bytes": 102400,
}


GPU_DEVICE_LIMITS = {
    "a100": {
        "warp_size": 32,
        "max_threads_per_sm": 2048,
        "max_blocks_per_sm": 32,
        "registers_per_sm": 65536,
        "shared_memory_per_sm_bytes": 163840,
    },
    "h100": {
        "warp_size": 32,
        "max_threads_per_sm": 2048,
        "max_blocks_per_sm": 32,
        "registers_per_sm": 65536,
        "shared_memory_per_sm_bytes": 233472,
    },
    "l4": {
        "warp_size": 32,
        "max_threads_per_sm": 1536,
        "max_blocks_per_sm": 24,
        "registers_per_sm": 65536,
        "shared_memory_per_sm_bytes": 102400,
    },
    "t4": {
        "warp_size": 32,
        "max_threads_per_sm": 1024,
        "max_blocks_per_sm": 16,
        "registers_per_sm": 65536,
        "shared_memory_per_sm_bytes": 65536,
    },
}


@dataclass(slots=True)
class KernelLaunchSummary:
    kernel_name: str
    registers_per_thread: int | None = None
    shared_memory_per_block_bytes: int | None = None
    threads_per_block: int | None = None
    block_dims: tuple[int, int, int] | None = None
    grid_dims: tuple[int, int, int] | None = None
    occupancy_estimate: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    dram_throughput_pct: float | None = None
    tensor_core_utilization_pct: float | None = None
    l1_cache_hit_rate_pct: float | None = None
    l2_cache_hit_rate_pct: float | None = None
    issue_slot_utilization_pct: float | None = None
    achieved_occupancy_pct: float | None = None
    eligible_warps_per_scheduler: float | None = None
    scheduler_active_pct: float | None = None
    dominant_warp_stall: str | None = None
    dominant_warp_stall_pct: float | None = None
    warp_stall_breakdown: dict = field(default_factory=dict)
    source: str = "kernel_inspector"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_occupancy(
    *,
    registers_per_thread: int | None,
    shared_memory_per_block_bytes: int | None,
    threads_per_block: int | None,
    device_limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    limits = {**DEFAULT_DEVICE_LIMITS, **(device_limits or {})}
    warp_size = max(1, int(limits["warp_size"]))
    max_threads = max(1, int(limits["max_threads_per_sm"]))
    max_blocks = max(1, int(limits["max_blocks_per_sm"]))
    registers_per_sm = max(1, int(limits["registers_per_sm"]))
    shared_per_sm = max(1, int(limits["shared_memory_per_sm_bytes"]))

    if not threads_per_block or threads_per_block <= 0:
        return {
            "occupancy_pct": None,
            "active_blocks_per_sm": None,
            "active_warps_per_sm": None,
            "limiting_factors": ["unknown_threads_per_block"],
            "device_limits": limits,
        }

    blocks_by_threads = max_threads // threads_per_block
    candidates = {
        "threads": blocks_by_threads,
        "blocks": max_blocks,
    }

    if registers_per_thread and registers_per_thread > 0:
        registers_per_block = registers_per_thread * threads_per_block
        candidates["registers"] = registers_per_sm // max(1, registers_per_block)

    if shared_memory_per_block_bytes and shared_memory_per_block_bytes > 0:
        candidates["shared_memory"] = shared_per_sm // shared_memory_per_block_bytes

    active_blocks = max(0, min(candidates.values()))
    active_threads = active_blocks * threads_per_block
    active_warps = active_threads // warp_size
    occupancy = min(active_threads / max_threads, 1.0)
    limiting_value = min(candidates.values())
    limiting = sorted(name for name, value in candidates.items() if value == limiting_value)

    return {
        "occupancy_pct": round(occupancy * 100.0, 2),
        "active_blocks_per_sm": active_blocks,
        "active_warps_per_sm": active_warps,
        "limiting_factors": limiting,
        "blocks_per_sm_limits": candidates,
        "device_limits": limits,
    }


def device_limits_for_gpu(gpu_model: str | None) -> dict[str, int]:
    if not gpu_model:
        return dict(DEFAULT_DEVICE_LIMITS)
    normalized = gpu_model.lower().replace("nvidia", "").replace("tesla", "").strip()
    for key, limits in GPU_DEVICE_LIMITS.items():
        if key in normalized:
            return {**DEFAULT_DEVICE_LIMITS, **limits}
    return dict(DEFAULT_DEVICE_LIMITS)


def launch_summary_from_attrs(
    attrs: dict[str, Any],
    *,
    device_limits: dict[str, int] | None = None,
    source: str = "kernel_attrs",
) -> KernelLaunchSummary:
    block_dims = _dims_from_attrs(attrs, ("block_x", "block_y", "block_z"), "block_dims")
    grid_dims = _dims_from_attrs(attrs, ("grid_x", "grid_y", "grid_z"), "grid_dims")
    threads_per_block = _positive_int(attrs.get("threads_per_block"))
    if threads_per_block is None and block_dims is not None:
        threads_per_block = block_dims[0] * block_dims[1] * block_dims[2]

    registers = _positive_int(attrs.get("registers_per_thread"))
    shared = _positive_int(attrs.get("shared_memory_per_block_bytes"))
    return KernelLaunchSummary(
        kernel_name=str(attrs.get("kernel_name_raw") or attrs.get("kernel_name") or "unknown"),
        registers_per_thread=registers,
        shared_memory_per_block_bytes=shared,
        threads_per_block=threads_per_block,
        block_dims=block_dims,
        grid_dims=grid_dims,
        occupancy_estimate=estimate_occupancy(
            registers_per_thread=registers,
            shared_memory_per_block_bytes=shared,
            threads_per_block=threads_per_block,
            device_limits=device_limits,
        ),
        source=source,
    )


def parse_nsight_compute_csv(path: str | Path) -> list[KernelLaunchSummary]:
    rows = _read_csv_rows(path)
    return _rows_to_kernel_summaries(rows)


def parse_nsight_compute_csv_text(text: str) -> list[KernelLaunchSummary]:
    rows = _text_to_csv_rows(text)
    return _rows_to_kernel_summaries(rows)


def _rows_to_kernel_summaries(rows: list[dict[str, str]]) -> list[KernelLaunchSummary]:
    by_kernel: dict[str, dict[str, Any]] = {}

    for row in rows:
        kernel = _first_present(
            row,
            "Kernel Name",
            "Kernel",
            "Kernel Name Demangled",
            "Name",
        )
        if not kernel:
            continue
        entry = by_kernel.setdefault(str(kernel), {"kernel_name_raw": str(kernel), "metrics": {}})

        metric_name = _first_present(row, "Metric Name", "Metric", "Name")
        metric_value = _first_present(row, "Metric Value", "Value", "Avg", "Average")
        if metric_name and metric_value is not None:
            canonical = _canonical_ncu_metric_name(str(metric_name))
            number = _to_float(metric_value)
            if number is not None:
                entry["metrics"][canonical] = number
                _apply_ncu_metric(entry, canonical, number)

        for column, key in _DIRECT_NCU_COLUMNS.items():
            value = _to_float(row.get(column))
            if value is not None:
                entry[key] = int(value)

    summaries: list[KernelLaunchSummary] = []
    for entry in by_kernel.values():
        summary = launch_summary_from_attrs(entry, source="nsight_compute_csv")
        summary.metrics = dict(entry.get("metrics", {}))
        _compute_derived_ncu_fields(summary)
        summaries.append(summary)
    return summaries


def map_nsight_compute_csv_to_ir(
    path: str | Path,
    *,
    run_id: str,
    device_id: str = "gpu0",
) -> tuple[list[EventRecord], list[MetricRecord]]:
    events: list[EventRecord] = []
    metrics: list[MetricRecord] = []
    for index, summary in enumerate(parse_nsight_compute_csv(path)):
        attrs = summary.to_dict()
        events.append(
            EventRecord(
                event_id=f"ncu_kernel_{index}",
                run_id=run_id,
                event_family="kernel",
                event_type="cuda_kernel_static_metrics",
                ts_start_ns=0,
                ts_end_ns=0,
                duration_ns=0,
                source="nsight_compute",
                device_id=device_id,
                attrs=attrs,
            )
        )
        for metric_name, value in summary.metrics.items():
            metrics.append(
                MetricRecord(
                    metric_id=f"ncu_metric_{index}_{metric_name}",
                    run_id=run_id,
                    metric_name=f"kernel.{metric_name}",
                    metric_unit=_metric_unit(metric_name),
                    value=value,
                    ts_ns=0,
                    source="nsight_compute",
                    device_id=device_id,
                    attrs={"kernel_name": summary.kernel_name},
                )
            )
    return events, metrics


def summarize_kernel_launches(events: list[EventRecord]) -> dict[str, Any]:
    summaries = [
        launch_summary_from_attrs(event.attrs).to_dict()
        for event in events
        if event.event_family == "kernel"
    ]
    rich = [
        item for item in summaries
        if item.get("registers_per_thread") is not None
        or item.get("shared_memory_per_block_bytes") is not None
        or item.get("threads_per_block") is not None
    ]
    occupancy_values = [
        float(item["occupancy_estimate"]["occupancy_pct"])
        for item in rich
        if isinstance(item.get("occupancy_estimate"), dict)
        and item["occupancy_estimate"].get("occupancy_pct") is not None
    ]
    return {
        "kernel_count": len(summaries),
        "kernels_with_launch_metadata": len(rich),
        "average_estimated_occupancy_pct": (
            round(sum(occupancy_values) / len(occupancy_values), 2)
            if occupancy_values else None
        ),
        "kernels": rich,
    }


def extract_cuda_binary_text(
    binary_path: str | Path,
    *,
    output: str,
    tool: str | None = None,
    timeout_s: int = 30,
) -> str:
    binary = Path(binary_path)
    if output not in {"ptx", "sass"}:
        raise ValueError("output must be 'ptx' or 'sass'")
    selected_tool = tool or ("cuobjdump" if output == "ptx" else "nvdisasm")
    executable = shutil.which(selected_tool)
    if executable is None:
        raise FileNotFoundError(f"{selected_tool} was not found on PATH")

    command = _cuda_extract_command(executable, binary, output)
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
    return result.stdout


def _cuda_extract_command(executable: str, binary: Path, output: str) -> list[str]:
    name = Path(executable).name.lower()
    if "cuobjdump" in name:
        flag = "--dump-ptx" if output == "ptx" else "--dump-sass"
        return [executable, flag, str(binary)]
    if "nvdisasm" in name:
        return [executable, "-c", str(binary)]
    raise ValueError(f"unsupported CUDA extraction tool: {executable}")


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    return _text_to_csv_rows(text)


def _text_to_csv_rows(text: str) -> list[dict[str, str]]:
    lines = [
        line for line in text.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and not line.lstrip().startswith("==")  # strip ==PROF== metadata from NCU exports
    ]
    if not lines:
        return []
    return list(csv.DictReader(lines))


def _compute_derived_ncu_fields(summary: KernelLaunchSummary) -> None:
    m = summary.metrics
    summary.dram_throughput_pct = m.get("dram_throughput_pct")
    summary.tensor_core_utilization_pct = m.get("tensor_core_utilization_pct")
    summary.l1_cache_hit_rate_pct = m.get("l1_cache_hit_rate_pct")
    summary.l2_cache_hit_rate_pct = m.get("l2_cache_hit_rate_pct")
    summary.issue_slot_utilization_pct = m.get("issue_slot_utilization_pct")
    summary.achieved_occupancy_pct = m.get("achieved_occupancy_pct")
    summary.eligible_warps_per_scheduler = m.get("eligible_warps_per_scheduler")
    summary.scheduler_active_pct = m.get("scheduler_active_pct")
    stalls = {k[len("warp_stall_"):]: v for k, v in m.items() if k.startswith("warp_stall_")}
    summary.warp_stall_breakdown = stalls
    if stalls:
        dominant_key = max(stalls, key=stalls.__getitem__)
        summary.dominant_warp_stall = dominant_key
        summary.dominant_warp_stall_pct = stalls[dominant_key]


def _apply_ncu_metric(entry: dict[str, Any], metric_name: str, value: float) -> None:
    if metric_name == "registers_per_thread":
        entry["registers_per_thread"] = int(value)
    elif metric_name == "shared_memory_per_block_bytes":
        entry["shared_memory_per_block_bytes"] = int(entry.get("shared_memory_per_block_bytes", 0)) + int(value)
    elif metric_name == "threads_per_block":
        entry["threads_per_block"] = int(value)
    elif metric_name in {"block_x", "block_y", "block_z", "grid_x", "grid_y", "grid_z"}:
        entry[metric_name] = int(value)


_NCU_STALL_PREFIX = "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_"


def _canonical_ncu_metric_name(name: str) -> str:
    lowered = name.strip().lower()
    lowered = lowered.replace(" ", "_").replace("-", "_").replace(".", "_")
    aliases = {
        # Launch config
        "launch__registers_per_thread": "registers_per_thread",
        "registers_per_thread": "registers_per_thread",
        "launch__shared_mem_per_block_static": "shared_memory_per_block_bytes",
        "launch__shared_mem_per_block_dynamic": "shared_memory_per_block_bytes",
        "shared_memory_per_block": "shared_memory_per_block_bytes",
        "shared_memory_per_block_bytes": "shared_memory_per_block_bytes",
        "launch__block_size": "threads_per_block",
        "threads_per_block": "threads_per_block",
        "launch__block_dim_x": "block_x",
        "launch__block_dim_y": "block_y",
        "launch__block_dim_z": "block_z",
        "launch__grid_dim_x": "grid_x",
        "launch__grid_dim_y": "grid_y",
        "launch__grid_dim_z": "grid_z",
        # Occupancy
        "sm__warps_active_avg_pct_of_peak_sustained_active": "achieved_occupancy_pct",
        "sm__warps_active_avg_pct_of_peak_sustained_elapsed": "achieved_occupancy_pct",
        "smsp__warps_eligible_avg_per_cycle_active": "eligible_warps_per_scheduler",
        "eligible_warps_per_scheduler": "eligible_warps_per_scheduler",
        "smsp__warps_active_avg_pct_of_peak_sustained_active": "scheduler_active_pct",
        "smsp__warps_active_avg_pct_of_peak_sustained_elapsed": "scheduler_active_pct",
        "scheduler_active_pct": "scheduler_active_pct",
        # DRAM / memory bandwidth
        "dram__throughput_avg_pct_of_peak_sustained_elapsed": "dram_throughput_pct",
        "memory_throughput": "dram_throughput_pct",
        "memory throughput": "dram_throughput_pct",
        # Tensor core utilization
        "sm__pipe_tensor_cycles_active_avg_pct_of_peak_sustained_active": "tensor_core_utilization_pct",
        "sm__pipe_tensor_cycles_active_avg_pct_of_peak_sustained_elapsed": "tensor_core_utilization_pct",
        "tensor_core_utilization": "tensor_core_utilization_pct",
        "tensor core utilization": "tensor_core_utilization_pct",
        # Cache hit rates
        "l1tex__t_sector_hit_rate_pct": "l1_cache_hit_rate_pct",
        "l1/tex_cache_throughput": "l1_cache_hit_rate_pct",
        "lts__t_sector_hit_rate_pct": "l2_cache_hit_rate_pct",
        "l2_cache_hit_rate": "l2_cache_hit_rate_pct",
        # Issue slot utilization (IPC proxy)
        "sm__issue_active_avg_pct_of_peak_sustained_active": "issue_slot_utilization_pct",
        "sm__issue_active_avg_pct_of_peak_sustained_elapsed": "issue_slot_utilization_pct",
        "smsp__issue_active_avg_pct_of_peak_sustained_active": "issue_slot_utilization_pct",
        "smsp__issue_active_avg_pct_of_peak_sustained_elapsed": "issue_slot_utilization_pct",
        "issue_slots_busy": "issue_slot_utilization_pct",
        "issue slots busy": "issue_slot_utilization_pct",
    }
    if lowered in aliases:
        return aliases[lowered]
    # Regex-free fallback: arbitrary warp stall reasons from NCU sampling data
    if lowered.startswith(_NCU_STALL_PREFIX):
        stall_type = lowered[len(_NCU_STALL_PREFIX):]
        return f"warp_stall_{stall_type}"
    per_warp_prefix = "smsp__warp_issue_stalled_"
    per_warp_suffix = "_per_warp_active_pct"
    if lowered.startswith(per_warp_prefix) and lowered.endswith(per_warp_suffix):
        stall_type = lowered[len(per_warp_prefix):-len(per_warp_suffix)]
        return f"warp_stall_{stall_type}"
    return aliases.get(lowered, lowered)


_DIRECT_NCU_COLUMNS = {
    "Registers Per Thread": "registers_per_thread",
    "Shared Memory Per Block": "shared_memory_per_block_bytes",
    "Shared Memory Per Block Bytes": "shared_memory_per_block_bytes",
    "Threads Per Block": "threads_per_block",
    "Block X": "block_x",
    "Block Y": "block_y",
    "Block Z": "block_z",
    "Grid X": "grid_x",
    "Grid Y": "grid_y",
    "Grid Z": "grid_z",
}


def _dims_from_attrs(
    attrs: dict[str, Any],
    scalar_keys: tuple[str, str, str],
    tuple_key: str,
) -> tuple[int, int, int] | None:
    raw = attrs.get(tuple_key)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in raw.replace("x", ",").split(",")]
        raw = parsed
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        dims = tuple(_positive_int(value) or 1 for value in raw)
        return (dims[0], dims[1], dims[2])

    values = [_positive_int(attrs.get(key)) for key in scalar_keys]
    if values[0] is None:
        return None
    return (values[0], values[1] or 1, values[2] or 1)


def _positive_int(value: Any) -> int | None:
    number = _to_float(value)
    if number is None or number < 0:
        return None
    return int(number)


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _metric_unit(metric_name: str) -> str:
    if metric_name.endswith("_pct") or metric_name.startswith("warp_stall_"):
        return "percent"
    if metric_name.endswith("_bytes"):
        return "bytes"
    return "count"
