"""Cross-layer evidence reconciliation for Fournex.

Merges signals from source, PTX, NCU, and profiler layers into unified
diagnoses with calibrated confidence labels.
"""
from __future__ import annotations

from typing import Any


# ── Signal extractors ─────────────────────────────────────────────────────────

def _signals_from_static(static: dict) -> dict:
    codes = {f.get("code", "") for f in static.get("findings", [])}
    styles: set[str] = set()
    for k in static.get("kernels", []):
        styles.update(k.get("memory_access_styles", []))
    return {
        "finding_codes": codes,
        "access_styles": styles,
        "strided_or_pitched": "strided_or_pitched" in codes or "strided_or_pitched" in styles,
        "unnecessary_syncthreads": "unnecessary_syncthreads" in codes,
        "conditional_syncthreads": "conditional_syncthreads" in codes,
    }


def _signals_from_ptx(ptx: dict) -> dict:
    from .recommendations.signals import extract_ptx_signals
    return extract_ptx_signals(ptx.get("run_summary", {}), ptx.get("bottlenecks", []))


def _signals_from_ncu(ncu: dict) -> dict:
    from .recommendations.signals import extract_ncu_signals
    return extract_ncu_signals(ncu.get("ncu_run_summary", {}), ncu.get("bottlenecks", []))


def _signals_from_profiler(profiler: dict) -> dict:
    labels = {b.get("label", "") for b in profiler.get("bottlenecks", [])}
    return {
        "sync_bound": "sync_bound" in labels,
        "input_bound": "input_bound" in labels,
    }


# ── Confidence ─────────────────────────────────────────────────────────────────

def _compute_confidence(n_confirming: int, n_available: int) -> str:
    """Map (confirming, available) to a confidence label.

    | confirming | available | confidence   |
    |------------|-----------|--------------|
    | 1          | 1         | medium       |
    | 1          | 2+        | low-medium   |
    | 2          | 2         | high         |
    | 2          | 3+        | medium-high  |
    | 3+         | any       | confirmed    |
    """
    if n_confirming >= 3:
        return "confirmed"
    if n_confirming == 2:
        return "high" if n_available == 2 else "medium-high"
    return "medium" if n_available <= 1 else "low-medium"


def _missing_evidence_for(
    entry: dict[str, Any],
    layers_confirming: list[str],
    layers_available: list[str],
) -> dict[str, Any] | None:
    """Return missing-evidence block for a diagnosis, or None if nothing actionable is missing."""
    needed = entry.get("evidence_needed", {})
    metrics: list[dict[str, Any]] = []
    for layer, layer_metrics in needed.items():
        if layer not in layers_confirming:
            for m in layer_metrics:
                metrics.append({**m, "layer": layer})
    if not metrics:
        return None
    ncu_metrics = [m["metric"] for m in metrics if m["layer"] == "ncu"]
    n_confirming = len(layers_confirming)
    n_available = max(len(layers_available), n_confirming)
    return {
        "metrics": metrics,
        "ncu_command": (
            "ncu --metrics " + ",".join(ncu_metrics) + " --csv ./report.csv ./your_kernel"
            if ncu_metrics else None
        ),
        "full_collection_command": "ncu --set full --csv ./report.csv ./your_kernel",
        "confidence_if_confirmed": _compute_confidence(n_confirming + 1, n_available),
    }


# ── Diagnosis catalog ──────────────────────────────────────────────────────────

_CATALOG: list[dict[str, Any]] = [
    {
        "label": "inefficient_global_memory_access",
        "display_name": "Inefficient global memory access",
        "severity": "high",
        "fix_summary": "Tile data through shared memory; ensure stride-1 warp access",
        "recommendation_ids": ["rec_ncu_improve_coalescing", "rec_ncu_tiling_shared_mem"],
        # per-layer condition checks (None means the diagnosis has no signal for that layer)
        "source_check": lambda s: s.get("strided_or_pitched", False),
        "source_claims": frozenset({"strided_or_pitched"}),
        "source_style_claims": frozenset({"strided_or_pitched"}),
        "ptx_check": lambda s: bool(s.get("ptx_global_memory_heavy") or s.get("ptx_no_shared_memory")),
        "ptx_claims": frozenset({"ptx_global_memory_heavy"}),
        "ncu_check": lambda s: bool(s.get("uncoalesced_global_loads")),
        "ncu_claims": frozenset({"uncoalesced_access"}),
        "ncu_metric_keys": ("avg_global_load_sectors_per_request",),
        "profiler_check": None,
        "profiler_claims": frozenset(),
        "evidence_needed": {
            "ncu": [
                {
                    "metric": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum_per_request",
                    "label": "Global load sectors/request",
                    "why": "> 4 confirms non-coalesced warp loads (ideal = 1)",
                },
                {
                    "metric": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                    "label": "DRAM throughput %",
                    "why": "high % confirms memory bandwidth is a bottleneck",
                },
            ],
        },
    },
    {
        "label": "excessive_synchronization",
        "display_name": "Excessive synchronization",
        "severity": "medium",
        "fix_summary": "Remove spurious __syncthreads(); avoid barriers inside divergent branches",
        "recommendation_ids": [],
        "source_check": lambda s: bool(s.get("unnecessary_syncthreads") or s.get("conditional_syncthreads")),
        "source_claims": frozenset({"unnecessary_syncthreads", "conditional_syncthreads"}),
        "source_style_claims": frozenset(),
        "ptx_check": None,
        "ptx_claims": frozenset(),
        "ncu_check": lambda s: bool(s.get("warp_stall_is_sync")),
        "ncu_claims": frozenset({"warp_stall_sync"}),
        "ncu_metric_keys": ("memory_stall_fraction",),
        "profiler_check": lambda s: bool(s.get("sync_bound")),
        "profiler_claims": frozenset({"sync_bound"}),
        "evidence_needed": {
            "ncu": [
                {
                    "metric": "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
                    "label": "Warp stall on barrier %",
                    "why": "high % confirms warps are stalling at __syncthreads() barriers",
                },
            ],
        },
    },
    {
        "label": "register_pressure",
        "display_name": "Register pressure",
        "severity": "high",
        "fix_summary": "Use __launch_bounds__ to cap register use; restructure to reduce spills",
        "recommendation_ids": [],
        "source_check": None,
        "source_claims": frozenset(),
        "source_style_claims": frozenset(),
        "ptx_check": lambda s: bool(s.get("ptx_has_register_spills") or s.get("ptx_high_register_count")),
        "ptx_claims": frozenset({"ptx_register_spills", "ptx_register_pressure"}),
        "ncu_check": lambda s: bool(s.get("occupancy_limited_by_registers") and not s.get("occupancy_good")),
        "ncu_claims": frozenset({"occupancy_limited_by_registers"}),
        "ncu_metric_keys": ("avg_occupancy_pct",),
        "profiler_check": None,
        "profiler_claims": frozenset(),
        "evidence_needed": {
            "ncu": [
                {
                    "metric": "sm__warps_active.avg.pct_of_peak_sustained_active",
                    "label": "Achieved occupancy %",
                    "why": "low % confirms register pressure is limiting resident warps",
                },
                {
                    "metric": "launch__registers_per_thread",
                    "label": "Registers per thread",
                    "why": "> 64 is the typical threshold where occupancy starts dropping",
                },
            ],
        },
    },
    {
        "label": "tensor_core_underutilization",
        "display_name": "Tensor core underutilization",
        "severity": "high",
        "fix_summary": "Use FP16/BF16 inputs; verify batch dims are multiples of 16",
        "recommendation_ids": [],
        "source_check": None,
        "source_claims": frozenset(),
        "source_style_claims": frozenset(),
        "ptx_check": None,
        "ptx_claims": frozenset(),
        "ncu_check": lambda s: bool(s.get("tensor_core_underutilized")),
        "ncu_claims": frozenset({"tensor_core_underutilized"}),
        "ncu_metric_keys": ("tensor_core_utilization_pct",),
        "profiler_check": None,
        "profiler_claims": frozenset(),
        "evidence_needed": {
            "ncu": [
                {
                    "metric": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
                    "label": "Tensor pipe utilization %",
                    "why": "< 30% confirms tensor cores are idle despite eligible workload",
                },
                {
                    "metric": "smsp__inst_executed_pipe_tensor.sum",
                    "label": "HMMA instruction count",
                    "why": "near-zero confirms no HMMA instructions were issued",
                },
                {
                    "metric": "sm__warps_active.avg.pct_of_peak_sustained_active",
                    "label": "Achieved occupancy %",
                    "why": "low occupancy amplifies tensor core underutilization",
                },
                {
                    "metric": "smsp__inst_executed_pipe_fma.sum",
                    "label": "FP32 FMA instruction count",
                    "why": "high FMA vs HMMA ratio confirms the kernel is taking the FP32 path",
                },
            ],
        },
    },
    {
        "label": "memory_bandwidth_saturation",
        "display_name": "Memory bandwidth saturation",
        "severity": "high",
        "fix_summary": "Fuse kernels to reduce DRAM traffic; tile for L1/L2 reuse",
        "recommendation_ids": ["rec_ncu_memory_bandwidth_bound"],
        "source_check": None,
        "source_claims": frozenset(),
        "source_style_claims": frozenset(),
        "ptx_check": lambda s: bool(s.get("ptx_global_memory_heavy")),
        "ptx_claims": frozenset({"ptx_global_memory_heavy"}),
        "ncu_check": lambda s: bool(s.get("memory_bandwidth_saturated")),
        "ncu_claims": frozenset({"memory_bandwidth_bound"}),
        "ncu_metric_keys": ("dram_throughput_pct",),
        "profiler_check": None,
        "profiler_claims": frozenset(),
        "evidence_needed": {
            "ncu": [
                {
                    "metric": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                    "label": "DRAM throughput %",
                    "why": "> 80% confirms memory bandwidth is saturated",
                },
                {
                    "metric": "lts__t_bytes.sum.pct_of_peak_sustained_elapsed",
                    "label": "L2 cache throughput %",
                    "why": "high L2 % alongside high DRAM % confirms cache hierarchy pressure",
                },
            ],
        },
    },
    {
        "label": "warp_divergence_risk",
        "display_name": "Warp divergence risk",
        "severity": "medium",
        "fix_summary": "Minimize thread-divergent branches; restructure conditionals to be warp-uniform",
        "recommendation_ids": [],
        "source_check": None,
        "source_claims": frozenset(),
        "source_style_claims": frozenset(),
        "ptx_check": lambda s: bool(s.get("ptx_branch_divergence_risk")),
        "ptx_claims": frozenset({"ptx_high_branch_density"}),
        "ncu_check": None,
        "ncu_claims": frozenset(),
        "ncu_metric_keys": (),
        "profiler_check": None,
        "profiler_claims": frozenset(),
        "evidence_needed": {
            "ncu": [
                {
                    "metric": "smsp__thread_inst_executed_per_inst_executed.ratio",
                    "label": "Thread execution efficiency",
                    "why": "< 1.0 confirms threads in the same warp took different paths",
                },
                {
                    "metric": "smsp__inst_executed_op_branch.sum",
                    "label": "Branch instruction count",
                    "why": "high count relative to total instructions confirms branch-heavy code",
                },
            ],
        },
    },
    {
        "label": "roofline_memory_bound",
        "display_name": "Memory-bandwidth limited (Roofline model)",
        "severity": "high",
        "fix_summary": (
            "Arithmetic intensity is below the ridge point — kernel spends more time "
            "waiting for DRAM than computing. Increase data reuse: tile through shared "
            "memory, fuse adjacent kernels, or reduce redundant global loads."
        ),
        "recommendation_ids": ["rec_roofline_tiling", "rec_roofline_kernel_fusion"],
        "source_check": None,
        "source_claims": frozenset(),
        "source_style_claims": frozenset(),
        "ptx_check": None,
        "ptx_claims": frozenset(),
        "ncu_check": lambda s: s.get("roofline_region") == "memory_bound",
        "ncu_claims": frozenset({"roofline_memory_bound"}),
        "ncu_metric_keys": ("arithmetic_intensity", "achieved_tflops", "mfu_pct", "memory_utilization_pct"),
        "profiler_check": None,
        "profiler_claims": frozenset(),
        "evidence_needed": {
            "ncu": [
                {
                    "metric": "dram__bytes_read.sum",
                    "label": "DRAM bytes read",
                    "why": "total memory traffic volume (numerator of arithmetic intensity)",
                },
                {
                    "metric": "smsp__inst_executed_pipe_fma.sum",
                    "label": "FP32 FMA instruction count",
                    "why": "compute work (denominator of arithmetic intensity)",
                },
            ],
        },
    },
    {
        "label": "roofline_low_mfu",
        "display_name": "Low compute utilization (MFU < 20%)",
        "severity": "medium",
        "fix_summary": (
            "Achieved TFLOP/s is well below the compute ceiling. The kernel is not "
            "memory-bound but still under-utilizes the GPU. Common causes: small tile "
            "sizes, low parallelism, or a launch configuration that leaves most SMs idle."
        ),
        "recommendation_ids": ["rec_roofline_occupancy", "rec_roofline_batch_size"],
        "source_check": None,
        "source_claims": frozenset(),
        "source_style_claims": frozenset(),
        "ptx_check": None,
        "ptx_claims": frozenset(),
        "ncu_check": lambda s: (
            s.get("mfu_pct") is not None
            and s.get("mfu_pct") < 20.0
            and s.get("roofline_region") != "memory_bound"
        ),
        "ncu_claims": frozenset({"roofline_low_mfu"}),
        "ncu_metric_keys": ("mfu_pct", "achieved_tflops", "peak_tflops"),
        "profiler_check": None,
        "profiler_claims": frozenset(),
        "evidence_needed": {
            "ncu": [
                {
                    "metric": "sm__warps_active.avg.pct_of_peak_sustained_active",
                    "label": "Achieved occupancy %",
                    "why": "low occupancy explains gaps between the compute ceiling and achieved throughput",
                },
                {
                    "metric": "smsp__inst_executed_pipe_fma.sum",
                    "label": "FP32 FMA instruction count",
                    "why": "confirms how much compute work was actually issued",
                },
            ],
        },
    },
]

_LAYER_CHECK_KEY = {
    "source": "source_check",
    "ptx": "ptx_check",
    "ncu": "ncu_check",
    "profiler": "profiler_check",
}


# ── Public API ────────────────────────────────────────────────────────────────

def reconcile_evidence(
    *,
    static: dict | None = None,
    ptx: dict | None = None,
    ncu: dict | None = None,
    profiler: dict | None = None,
) -> dict[str, Any]:
    """Merge cross-layer signals into unified diagnoses with confidence labels.

    Parameters
    ----------
    static:   output of ``inspect_cuda_source()``
    ptx:      output of ``analyze_ptx_text()``
    ncu:      output of ``analyze_ncu_csv_text()``
    profiler: output of ``summarize_run()`` / ``summarize_step_scope()``

    Returns a dict with schema ``reconciliation_v1``.
    """
    layers_available: list[str] = []

    static_sigs: dict | None = _signals_from_static(static) if static is not None else None
    ptx_sigs: dict | None = _signals_from_ptx(ptx) if ptx is not None else None
    ncu_sigs: dict | None = _signals_from_ncu(ncu) if ncu is not None else None
    profiler_sigs: dict | None = _signals_from_profiler(profiler) if profiler is not None else None

    if static is not None:
        layers_available.append("source")
    if ptx is not None:
        layers_available.append("ptx")
    if ncu is not None:
        layers_available.append("ncu")
    if profiler is not None:
        layers_available.append("profiler")

    # Collect all existing codes/labels per layer for unreconciled tracking
    static_codes: set[str] = static_sigs.get("finding_codes", set()) if static_sigs else set()
    ptx_labels: set[str] = {b["label"] for b in (ptx or {}).get("bottlenecks", [])}
    ncu_labels: set[str] = {b["label"] for b in (ncu or {}).get("bottlenecks", [])}
    profiler_labels: set[str] = {b.get("label", "") for b in (profiler or {}).get("bottlenecks", [])}

    claimed_static: set[str] = set()
    claimed_ptx: set[str] = set()
    claimed_ncu: set[str] = set()
    claimed_profiler: set[str] = set()

    layer_sigs = {
        "source": static_sigs,
        "ptx": ptx_sigs,
        "ncu": ncu_sigs,
        "profiler": profiler_sigs,
    }

    diagnoses: list[dict[str, Any]] = []

    for entry in _CATALOG:
        confirming: list[str] = []
        evidence: dict[str, Any] = {"source": None, "ptx": None, "ncu": None, "profiler": None}

        # Source layer
        if static_sigs is not None and entry.get("source_check") is not None:
            if entry["source_check"](static_sigs):
                confirming.append("source")
                matched_codes = sorted(entry["source_claims"] & static_sigs["finding_codes"])
                matched_styles = sorted(entry["source_style_claims"] & static_sigs["access_styles"])
                ev: dict[str, Any] = {}
                if matched_codes:
                    ev["findings"] = matched_codes
                if matched_styles:
                    ev["access_styles"] = matched_styles
                evidence["source"] = ev if ev else {"triggered": True}

        # PTX layer
        if ptx_sigs is not None and entry.get("ptx_check") is not None:
            if entry["ptx_check"](ptx_sigs):
                confirming.append("ptx")
                matched_ptx = sorted(entry["ptx_claims"] & ptx_labels)
                evidence["ptx"] = {"bottlenecks": matched_ptx} if matched_ptx else {"triggered": True}

        # NCU layer
        if ncu_sigs is not None and entry.get("ncu_check") is not None:
            if entry["ncu_check"](ncu_sigs):
                confirming.append("ncu")
                matched_ncu = sorted(entry["ncu_claims"] & ncu_labels)
                metrics = {
                    k: ncu_sigs[k]
                    for k in entry.get("ncu_metric_keys", ())
                    if k in ncu_sigs and ncu_sigs[k] is not None
                }
                ev_ncu: dict[str, Any] = {"bottlenecks": matched_ncu}
                if metrics:
                    ev_ncu["metrics"] = metrics
                evidence["ncu"] = ev_ncu

        # Profiler layer
        if profiler_sigs is not None and entry.get("profiler_check") is not None:
            if entry["profiler_check"](profiler_sigs):
                confirming.append("profiler")
                matched_prof = sorted(entry["profiler_claims"] & profiler_labels)
                evidence["profiler"] = {"bottlenecks": matched_prof} if matched_prof else {"triggered": True}

        if not confirming:
            continue

        n_available = sum(
            1
            for layer, check_key in _LAYER_CHECK_KEY.items()
            if layer_sigs[layer] is not None and entry.get(check_key) is not None
        )

        confidence = _compute_confidence(len(confirming), n_available)
        diagnoses.append({
            "label": entry["label"],
            "display_name": entry["display_name"],
            "confidence": confidence,
            "severity": entry["severity"],
            "layers_confirming": confirming,
            "evidence": evidence,
            "fix_summary": entry["fix_summary"],
            "recommendation_ids": entry["recommendation_ids"],
            "missing_evidence": _missing_evidence_for(entry, confirming, layers_available),
        })

        if "source" in confirming:
            claimed_static.update(entry["source_claims"])
        if "ptx" in confirming:
            claimed_ptx.update(entry["ptx_claims"])
        if "ncu" in confirming:
            claimed_ncu.update(entry["ncu_claims"])
        if "profiler" in confirming:
            claimed_profiler.update(entry["profiler_claims"])

    # Unreconciled: codes/labels present in a layer but not claimed by any diagnosis
    unreconciled: dict[str, list[str]] = {}
    if static is not None:
        unc = sorted(static_codes - claimed_static)
        if unc:
            unreconciled["source"] = unc
    if ptx is not None:
        unc = sorted(ptx_labels - claimed_ptx)
        if unc:
            unreconciled["ptx"] = unc
    if ncu is not None:
        unc = sorted(ncu_labels - claimed_ncu)
        if unc:
            unreconciled["ncu"] = unc
    if profiler is not None:
        unc = sorted(profiler_labels - claimed_profiler)
        if unc:
            unreconciled["profiler"] = unc

    return {
        "schema": "reconciliation_v1",
        "layers_available": layers_available,
        "diagnoses": diagnoses,
        "unreconciled": unreconciled,
    }


def what_evidence_is_missing(
    *,
    static: dict | None = None,
    ptx: dict | None = None,
    ncu: dict | None = None,
    profiler: dict | None = None,
) -> list[dict[str, Any]]:
    """Reconcile all layers and return only diagnoses that have actionable missing evidence."""
    result = reconcile_evidence(static=static, ptx=ptx, ncu=ncu, profiler=profiler)
    return [d for d in result["diagnoses"] if d.get("missing_evidence")]
