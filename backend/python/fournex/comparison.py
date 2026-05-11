from __future__ import annotations

from typing import Any

from .cuda_static import inspect_cuda_source
from .ncu_analysis import analyze_ncu_csv_text
from .ptx_analysis import analyze_ptx_text


# ── Public API ────────────────────────────────────────────────────────────────

def compare_implementations(
    a_input: dict[str, Any],
    b_input: dict[str, Any],
) -> dict[str, Any]:
    """Compare two CUDA implementations across static, PTX, and profiler dimensions.

    Each input dict:
        label, cuda_source, cuda_filename, ptx, ptx_filename, ncu_csv, gpu_model
    All fields except label are optional (None → skip that analysis layer).
    """
    label_a = a_input.get("label", "a")
    label_b = b_input.get("label", "b")

    static_a = _run_static(a_input)
    static_b = _run_static(b_input)
    ptx_a    = _run_ptx(a_input)
    ptx_b    = _run_ptx(b_input)
    ncu_a    = _run_ncu(a_input)
    ncu_b    = _run_ncu(b_input)

    static_diff = _build_static_diff(static_a, static_b)
    ptx_diff    = _build_ptx_diff(ptx_a, ptx_b)
    ncu_diff    = _build_ncu_diff(ncu_a, ncu_b)
    scorecard   = _build_scorecard(static_a, ptx_a, ncu_a, static_b, ptx_b, ncu_b)
    verdict     = _build_verdict(scorecard, label_a, label_b)

    return {
        "schema": "comparison_v1",
        "label_a": label_a,
        "label_b": label_b,
        "data_availability": {
            "a": {
                "cuda_source": static_a is not None,
                "ptx": ptx_a is not None,
                "ncu_csv": ncu_a is not None,
            },
            "b": {
                "cuda_source": static_b is not None,
                "ptx": ptx_b is not None,
                "ncu_csv": ncu_b is not None,
            },
        },
        "static_diff": static_diff,
        "ptx_diff": ptx_diff,
        "ncu_diff": ncu_diff,
        "scorecard": scorecard,
        "verdict": verdict,
    }


# ── Analysis runners ──────────────────────────────────────────────────────────

def _run_static(inp: dict[str, Any]) -> dict[str, Any] | None:
    src = inp.get("cuda_source")
    if not src:
        return None
    return inspect_cuda_source(
        src,
        filename=inp.get("cuda_filename", "<memory>"),
        gpu_model=inp.get("gpu_model"),
    )


def _run_ptx(inp: dict[str, Any]) -> dict[str, Any] | None:
    ptx = inp.get("ptx")
    if not ptx:
        return None
    return analyze_ptx_text(ptx, filename=inp.get("ptx_filename", "kernel.ptx"))


def _run_ncu(inp: dict[str, Any]) -> dict[str, Any] | None:
    csv = inp.get("ncu_csv")
    if not csv:
        return None
    return analyze_ncu_csv_text(csv)


# ── Diff builders ─────────────────────────────────────────────────────────────

def _build_static_diff(static_a: dict | None, static_b: dict | None) -> dict[str, Any]:
    if static_a is None and static_b is None:
        return {"available": False}

    sa = static_a or {}
    sb = static_b or {}

    kernels_a = sa.get("kernels", [])
    kernels_b = sb.get("kernels", [])

    smem_a = sum(len(k.get("shared_memory", [])) for k in kernels_a)
    smem_b = sum(len(k.get("shared_memory", [])) for k in kernels_b)

    bank_a = any(
        any(s.get("bank_conflict_risk", False) for s in k.get("shared_memory", []))
        for k in kernels_a
    )
    bank_b = any(
        any(s.get("bank_conflict_risk", False) for s in k.get("shared_memory", []))
        for k in kernels_b
    )

    atomics_a = any(k.get("atomics") for k in kernels_a)
    atomics_b = any(k.get("atomics") for k in kernels_b)

    reductions_a = any(k.get("reductions") for k in kernels_a)
    reductions_b = any(k.get("reductions") for k in kernels_b)

    idx_a: set[str] = {p for k in kernels_a for p in k.get("indexing_patterns", [])}
    idx_b: set[str] = {p for k in kernels_b for p in k.get("indexing_patterns", [])}

    mem_a: set[str] = {s for k in kernels_a for s in k.get("memory_access_styles", [])}
    mem_b: set[str] = {s for k in kernels_b for s in k.get("memory_access_styles", [])}

    findings_a = sa.get("findings", [])
    findings_b = sb.get("findings", [])

    return {
        "available": True,
        "kernel_count":              _diff_numeric(sa.get("kernel_count"), sb.get("kernel_count"), higher_is_better=False),
        "launch_count":              _diff_numeric(sa.get("launch_count"), sb.get("launch_count"), higher_is_better=False),
        "shared_memory_alloc_count": _diff_numeric(smem_a, smem_b, higher_is_better=True),
        "has_bank_conflict_risk":    _diff_bool(bank_a, bank_b, better_when_false=True),
        "atomics_used":              _diff_bool(atomics_a, atomics_b, better_when_false=False),
        "reductions_used":           _diff_bool(reductions_a, reductions_b, better_when_false=False),
        "indexing_patterns": {
            "a": sorted(idx_a),
            "b": sorted(idx_b),
            "added_in_b":   sorted(idx_b - idx_a),
            "removed_in_b": sorted(idx_a - idx_b),
        },
        "memory_access_styles": {
            "a": sorted(mem_a),
            "b": sorted(mem_b),
            "added_in_b":   sorted(mem_b - mem_a),
            "removed_in_b": sorted(mem_a - mem_b),
        },
        "findings_diff": _diff_findings(findings_a, findings_b),
    }


def _build_ptx_diff(ptx_a: dict | None, ptx_b: dict | None) -> dict[str, Any]:
    if ptx_a is None and ptx_b is None:
        return {"available": False}

    # Aggregate across all kernels using run_summary + per-kernel totals
    sa = ptx_a.get("run_summary", {}) if ptx_a else {}
    sb = ptx_b.get("run_summary", {}) if ptx_b else {}
    ks_a = ptx_a.get("kernels", []) if ptx_a else []
    ks_b = ptx_b.get("kernels", []) if ptx_b else []

    def _sum(ks: list, field: str) -> int:
        return sum(k.get(field, 0) for k in ks)

    def _any_bool(ks: list, field: str) -> bool:
        return any(k.get(field, False) for k in ks)

    reg_a   = round(sa.get("avg_register_count", 0) or 0)
    reg_b   = round(sb.get("avg_register_count", 0) or 0)
    locmem_a = _sum(ks_a, "local_memory_bytes")
    locmem_b = _sum(ks_b, "local_memory_bytes")
    spill_a  = _any_bool(ks_a, "has_register_spills")
    spill_b  = _any_bool(ks_b, "has_register_spills")
    inst_a   = sa.get("total_instructions") or _sum(ks_a, "instruction_count")
    inst_b   = sb.get("total_instructions") or _sum(ks_b, "instruction_count")

    # Instruction mix diff (aggregate across all kernels)
    mix_a: dict[str, int] = {}
    for k in ks_a:
        for cat, cnt in k.get("instruction_mix", {}).items():
            mix_a[cat] = mix_a.get(cat, 0) + cnt
    mix_b: dict[str, int] = {}
    for k in ks_b:
        for cat, cnt in k.get("instruction_mix", {}).items():
            mix_b[cat] = mix_b.get(cat, 0) + cnt
    all_cats = sorted(set(mix_a) | set(mix_b))
    mix_diff = {
        cat: {
            "a": mix_a.get(cat, 0),
            "b": mix_b.get(cat, 0),
            "delta": mix_b.get(cat, 0) - mix_a.get(cat, 0),
        }
        for cat in all_cats
    }

    findings_a = (ptx_a or {}).get("findings", [])
    findings_b = (ptx_b or {}).get("findings", [])

    return {
        "available":               True,
        "register_count":          _diff_numeric(reg_a, reg_b, higher_is_better=False),
        "local_memory_bytes":      _diff_numeric(locmem_a, locmem_b, higher_is_better=False),
        "has_register_spills":     _diff_bool(spill_a, spill_b, better_when_false=True),
        "instruction_count":       _diff_numeric(inst_a, inst_b, higher_is_better=False),
        "global_load_count":       _diff_numeric(_sum(ks_a, "global_load_count"), _sum(ks_b, "global_load_count"), higher_is_better=False),
        "global_store_count":      _diff_numeric(_sum(ks_a, "global_store_count"), _sum(ks_b, "global_store_count"), higher_is_better=False),
        "shared_load_count":       _diff_numeric(_sum(ks_a, "shared_load_count"), _sum(ks_b, "shared_load_count"), higher_is_better=True),
        "shared_store_count":      _diff_numeric(_sum(ks_a, "shared_store_count"), _sum(ks_b, "shared_store_count"), higher_is_better=True),
        "spill_load_count":        _diff_numeric(sa.get("total_spill_loads", 0) or _sum(ks_a, "spill_load_count"), sb.get("total_spill_loads", 0) or _sum(ks_b, "spill_load_count"), higher_is_better=False),
        "spill_store_count":       _diff_numeric(sa.get("total_spill_stores", 0) or _sum(ks_a, "spill_store_count"), sb.get("total_spill_stores", 0) or _sum(ks_b, "spill_store_count"), higher_is_better=False),
        "branch_count":            _diff_numeric(_sum(ks_a, "branch_count"), _sum(ks_b, "branch_count"), higher_is_better=False),
        "conditional_branch_count":_diff_numeric(_sum(ks_a, "conditional_branch_count"), _sum(ks_b, "conditional_branch_count"), higher_is_better=False),
        "instruction_mix_diff":    mix_diff,
        "findings_diff":           _diff_findings(findings_a, findings_b),
    }


def _build_ncu_diff(ncu_a: dict | None, ncu_b: dict | None) -> dict[str, Any]:
    if ncu_a is None and ncu_b is None:
        return {"available": False}

    rs_a = (ncu_a or {}).get("ncu_run_summary", {})
    rs_b = (ncu_b or {}).get("ncu_run_summary", {})

    def _get(d: dict, key: str) -> float | None:
        v = d.get(key)
        return float(v) if v is not None else None

    return {
        "available": True,
        "avg_dram_throughput_pct":         _diff_numeric(_get(rs_a, "avg_dram_throughput_pct"), _get(rs_b, "avg_dram_throughput_pct"), higher_is_better=False),
        "avg_l1_cache_hit_rate_pct":       _diff_numeric(_get(rs_a, "avg_l1_cache_hit_rate_pct"), _get(rs_b, "avg_l1_cache_hit_rate_pct"), higher_is_better=True),
        "avg_l2_cache_hit_rate_pct":       _diff_numeric(_get(rs_a, "avg_l2_cache_hit_rate_pct"), _get(rs_b, "avg_l2_cache_hit_rate_pct"), higher_is_better=True),
        "avg_issue_slot_utilization_pct":  _diff_numeric(_get(rs_a, "avg_issue_slot_utilization_pct"), _get(rs_b, "avg_issue_slot_utilization_pct"), higher_is_better=True),
        "avg_occupancy_pct":               _diff_numeric(_get(rs_a, "avg_occupancy_pct"), _get(rs_b, "avg_occupancy_pct"), higher_is_better=True),
        "avg_tensor_core_utilization_pct": _diff_numeric(_get(rs_a, "avg_tensor_core_utilization_pct"), _get(rs_b, "avg_tensor_core_utilization_pct"), higher_is_better=True),
        "memory_stall_fraction":           _diff_numeric(_get(rs_a, "memory_stall_fraction"), _get(rs_b, "memory_stall_fraction"), higher_is_better=False),
        "compute_stall_fraction":          _diff_numeric(_get(rs_a, "compute_stall_fraction"), _get(rs_b, "compute_stall_fraction"), higher_is_better=False),
        "dominant_warp_stall": {
            "a": rs_a.get("dominant_warp_stall"),
            "b": rs_b.get("dominant_warp_stall"),
        },
        "primary_bottleneck": {
            "a": (ncu_a or {}).get("primary_bottleneck"),
            "b": (ncu_b or {}).get("primary_bottleneck"),
        },
    }


# ── Scorecard ─────────────────────────────────────────────────────────────────

def _score_register_efficiency(ptx: dict | None) -> float | None:
    if ptx is None:
        return None
    ks = ptx.get("kernels", [])
    if not ks:
        return None
    # Use worst kernel (highest register count, any spill)
    spills = any(k.get("has_register_spills", False) for k in ks)
    avg_regs = ptx.get("run_summary", {}).get("avg_register_count") or 0
    reg_score = _clamp(1.0 - (avg_regs - 32) / 96, 0.0, 1.0)
    spill_mult = 0.0 if spills else 1.0
    return _clamp(spill_mult * reg_score, 0.0, 1.0)


def _score_memory_efficiency(ptx: dict | None, ncu: dict | None) -> float | None:
    # NCU path takes precedence
    if ncu is not None:
        rs = ncu.get("ncu_run_summary", {})
        dram = rs.get("avg_dram_throughput_pct")
        l1   = rs.get("avg_l1_cache_hit_rate_pct")
        l2   = rs.get("avg_l2_cache_hit_rate_pct")
        dram_score = _clamp(1.0 - (dram or 0) / 100, 0.0, 1.0) if dram is not None else None
        l1_score   = _clamp((l1 or 0) / 100, 0.0, 1.0) if l1 is not None else None
        l2_score   = _clamp((l2 or 0) / 100, 0.0, 1.0) if l2 is not None else None
        result = _weighted_avg_present([
            (dram_score, 0.30),
            (l1_score,   0.40),
            (l2_score,   0.30),
        ])
        if result is not None:
            return result

    # PTX fallback
    if ptx is not None:
        ks = ptx.get("kernels", [])
        if ks:
            total_inst  = sum(k.get("instruction_count", 0) for k in ks)
            global_load = sum(k.get("global_load_count", 0) for k in ks)
            global_store= sum(k.get("global_store_count", 0) for k in ks)
            shared_load = sum(k.get("shared_load_count", 0) for k in ks)
            shared_store= sum(k.get("shared_store_count", 0) for k in ks)
            denom = max(total_inst, 1)
            global_ratio = (global_load + global_store) / denom
            shared_ratio = (shared_load + shared_store) / denom
            global_penalty = _clamp(global_ratio / 0.40, 0.0, 1.0)
            shared_bonus   = _clamp(shared_ratio / 0.20, 0.0, 0.20)
            return _clamp(1.0 - global_penalty + shared_bonus, 0.0, 1.0)

    return None


def _score_compute_efficiency(ptx: dict | None, ncu: dict | None) -> float | None:
    # NCU path (preferred)
    if ncu is not None:
        rs = ncu.get("ncu_run_summary", {})
        isu = rs.get("avg_issue_slot_utilization_pct")
        occ = rs.get("avg_occupancy_pct")
        isu_score = _clamp((isu or 0) / 100, 0.0, 1.0) if isu is not None else None
        occ_score = _clamp((occ or 0) / 100, 0.0, 1.0) if occ is not None else None
        result = _weighted_avg_present([
            (isu_score, 0.60),
            (occ_score, 0.40),
        ])
        if result is not None:
            return result

    # PTX fallback — divergence proxy
    if ptx is not None:
        ks = ptx.get("kernels", [])
        if ks:
            total_inst = sum(k.get("instruction_count", 0) for k in ks)
            cond_branches = sum(k.get("conditional_branch_count", 0) for k in ks)
            branch_ratio = cond_branches / max(total_inst, 1)
            return _clamp(1.0 - branch_ratio / 0.15, 0.0, 1.0)

    return None


def _score_launch_efficiency(static: dict | None, ncu: dict | None) -> float | None:
    # NCU occupancy overrides static heuristic
    if ncu is not None:
        rs = ncu.get("ncu_run_summary", {})
        occ = rs.get("avg_occupancy_pct")
        if occ is not None:
            return _clamp(occ / 100, 0.0, 1.0)

    if static is not None:
        kernels = static.get("kernels", [])
        findings = static.get("findings", [])
        codes = {f.get("code", "") for f in findings}
        bank_penalty  = 0.20 if any(
            any(s.get("bank_conflict_risk", False) for s in k.get("shared_memory", []))
            for k in kernels
        ) else 0.0
        cond_sync_pen = 0.30 if "conditional_syncthreads" in codes else 0.0
        bounds_pen    = 0.10 if "missing_obvious_bounds_guard" in codes else 0.0
        return _clamp(1.0 - bank_penalty - cond_sync_pen - bounds_pen, 0.0, 1.0)

    return None


def _build_scorecard(
    static_a: dict | None, ptx_a: dict | None, ncu_a: dict | None,
    static_b: dict | None, ptx_b: dict | None, ncu_b: dict | None,
) -> dict[str, Any]:
    score_a_reg   = _score_register_efficiency(ptx_a)
    score_b_reg   = _score_register_efficiency(ptx_b)
    score_a_mem   = _score_memory_efficiency(ptx_a, ncu_a)
    score_b_mem   = _score_memory_efficiency(ptx_b, ncu_b)
    score_a_comp  = _score_compute_efficiency(ptx_a, ncu_a)
    score_b_comp  = _score_compute_efficiency(ptx_b, ncu_b)
    score_a_launch= _score_launch_efficiency(static_a, ncu_a)
    score_b_launch= _score_launch_efficiency(static_b, ncu_b)

    def _dim(score_a: float | None, score_b: float | None, weight: float) -> dict:
        avail = score_a is not None or score_b is not None
        sa = score_a if score_a is not None else 0.0
        sb = score_b if score_b is not None else 0.0
        if avail and abs(sb - sa) > _TIE:
            winner = "b" if sb > sa else "a"
        elif avail:
            winner = "tie"
        else:
            winner = None
        return {
            "score_a":   round(sa, 4) if score_a is not None else None,
            "score_b":   round(sb, 4) if score_b is not None else None,
            "weight":    weight,
            "available": avail,
            "winner":    winner,
        }

    return {
        "register_efficiency": _dim(score_a_reg,    score_b_reg,    0.20),
        "memory_efficiency":   _dim(score_a_mem,    score_b_mem,    0.30),
        "compute_efficiency":  _dim(score_a_comp,   score_b_comp,   0.30),
        "launch_efficiency":   _dim(score_a_launch, score_b_launch, 0.20),
    }


def _build_verdict(scorecard: dict[str, Any], label_a: str, label_b: str) -> dict[str, Any]:
    available = [(dim, d) for dim, d in scorecard.items() if d["available"]]
    if not available:
        return {
            "overall_winner": "tie",
            "score_a": None,
            "score_b": None,
            "score_delta": None,
            "dimensions_won_by_a": [],
            "dimensions_won_by_b": [],
        }

    total_weight = sum(d["weight"] for _, d in available)
    score_a = sum((d["score_a"] or 0.0) * d["weight"] for _, d in available) / total_weight
    score_b = sum((d["score_b"] or 0.0) * d["weight"] for _, d in available) / total_weight
    score_a = round(score_a, 4)
    score_b = round(score_b, 4)
    delta   = round(score_b - score_a, 4)

    dims_won_a = [dim for dim, d in available if d["winner"] == "a"]
    dims_won_b = [dim for dim, d in available if d["winner"] == "b"]

    if abs(score_b - score_a) <= _TIE:
        winner = "tie"
    elif score_b > score_a:
        winner = "b"
    else:
        winner = "a"

    return {
        "overall_winner":    winner,
        "score_a":           score_a,
        "score_b":           score_b,
        "score_delta":       delta,
        "dimensions_won_by_a": dims_won_a,
        "dimensions_won_by_b": dims_won_b,
    }


# ── Low-level helpers ─────────────────────────────────────────────────────────

_TIE = 0.02


def _diff_numeric(
    val_a: float | int | None,
    val_b: float | int | None,
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    if val_a is None and val_b is None:
        return {"a": None, "b": None, "delta": None, "better": None}
    delta = (val_b - val_a) if val_a is not None and val_b is not None else None
    if delta is None:
        better = None
    elif delta == 0:
        better = "tie"
    elif higher_is_better:
        better = "b" if delta > 0 else "a"
    else:
        better = "b" if delta < 0 else "a"
    return {"a": val_a, "b": val_b, "delta": delta, "better": better}


def _diff_bool(val_a: bool | None, val_b: bool | None, *, better_when_false: bool) -> dict[str, Any]:
    resolved = bool(val_a) and not bool(val_b)
    introduced = not bool(val_a) and bool(val_b)
    return {"a": val_a, "b": val_b, "resolved_in_b": resolved, "introduced_in_b": introduced}


def _diff_findings(
    findings_a: list[dict], findings_b: list[dict]
) -> dict[str, list[str]]:
    codes_a = {f.get("code", "") for f in findings_a if f.get("code")}
    codes_b = {f.get("code", "") for f in findings_b if f.get("code")}
    return {
        "resolved_in_b": sorted(codes_a - codes_b),
        "new_in_b":      sorted(codes_b - codes_a),
        "shared":        sorted(codes_a & codes_b),
    }


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _weighted_avg_present(pairs: list[tuple[float | None, float]]) -> float | None:
    present = [(v, w) for v, w in pairs if v is not None]
    if not present:
        return None
    total_w = sum(w for _, w in present)
    return sum(v * w for v, w in present) / total_w
