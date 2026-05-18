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
        "ncu_check": lambda s: bool(s.get("occupancy_limited_by_registers")),
        "ncu_claims": frozenset({"occupancy_limited_by_registers"}),
        "ncu_metric_keys": ("avg_occupancy_pct",),
        "profiler_check": None,
        "profiler_claims": frozenset(),
    },
    {
        "label": "tensor_core_underutilization",
        "display_name": "Tensor core underutilization",
        "severity": "medium",
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

        diagnoses.append({
            "label": entry["label"],
            "display_name": entry["display_name"],
            "confidence": _compute_confidence(len(confirming), n_available),
            "severity": entry["severity"],
            "layers_confirming": confirming,
            "evidence": evidence,
            "fix_summary": entry["fix_summary"],
            "recommendation_ids": entry["recommendation_ids"],
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
