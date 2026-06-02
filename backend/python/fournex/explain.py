"""frx explain -LLM-ready optimization brief from profiler output.

Produces three artifacts from an NCU CSV (+ optional CUDA source):
  frx_summary.txt      Human-readable root-cause narrative
  frx_llm_prompt.txt   Paste-ready prompt with evidence + guardrails
  frx_evidence.json    Structured metrics for tools/agents
"""
from __future__ import annotations

import json
import textwrap
from typing import Any

from .reconciliation import reconcile_evidence, what_evidence_is_missing

# ── Confidence / severity sort order ─────────────────────────────────────────

_CONFIDENCE_RANK: dict[str, int] = {
    "confirmed": 0, "high": 1, "medium-high": 2, "medium": 3, "low-medium": 4,
}
_SEVERITY_RANK: dict[str, int] = {"high": 0, "medium": 1}

# ── Metric display config ─────────────────────────────────────────────────────

# key → (label, unit, higher_is_better, good_threshold, bad_threshold)
_METRIC_CONFIG: dict[str, tuple[str, str, bool, float, float]] = {
    "avg_dram_throughput_pct":           ("DRAM Throughput",          "%", False, 50.0, 70.0),
    "avg_tensor_core_utilization_pct":   ("Tensor Core Util",         "%", True,  50.0, 20.0),
    "avg_l1_cache_hit_rate_pct":         ("L1 Cache Hit Rate",        "%", True,  70.0, 40.0),
    "avg_l2_cache_hit_rate_pct":         ("L2 Cache Hit Rate",        "%", True,  70.0, 50.0),
    "avg_global_load_sectors_per_request": ("Load Sectors/Request",   "",  False, 2.0,  4.0),
    "avg_issue_slot_utilization_pct":    ("Issue Utilization",        "%", True,  60.0, 40.0),
    "avg_occupancy_pct":                 ("Occupancy",                "%", True,  50.0, 30.0),
    "memory_stall_fraction":             ("Memory Stall Fraction",    "",  False, 0.2,  0.5),
}


def _metric_status(key: str, value: float) -> str:
    cfg = _METRIC_CONFIG.get(key)
    if cfg is None:
        return ""
    _, _, higher_better, good_thresh, bad_thresh = cfg
    if higher_better:
        if value >= good_thresh:
            return "OK"
        if value >= bad_thresh:
            return "LOW"
        return "POOR"
    else:
        if value <= good_thresh:
            return "OK"
        if value <= bad_thresh:
            return "ELEVATED"
        # Special label for throughput vs fraction
        if key == "avg_dram_throughput_pct":
            return "SATURATED"
        if key == "memory_stall_fraction":
            return "HIGH"
        return "POOR"


def _status_note(key: str, value: float) -> str:
    """Short contextual note for a metric value (used in evidence list)."""
    status = _metric_status(key, value)
    notes = {
        "avg_dram_throughput_pct": {
            "SATURATED": "saturated (>70%)", "ELEVATED": "elevated (>50%)", "OK": "ok (<50%)"
        },
        "avg_l1_cache_hit_rate_pct": {
            "POOR": "poor (<40%)", "LOW": "low (<70%)", "OK": "ok (>70%)"
        },
        "avg_l2_cache_hit_rate_pct": {
            "POOR": "poor (<50%)", "LOW": "low (<70%)", "OK": "ok (>70%)"
        },
        "avg_global_load_sectors_per_request": {
            "POOR": "poor (ideal: 1.0)", "ELEVATED": "elevated (ideal: 1.0)", "OK": "ok"
        },
        "avg_issue_slot_utilization_pct": {
            "POOR": "poor (<40%)", "LOW": "low (<60%)", "OK": "ok (>60%)"
        },
        "avg_occupancy_pct": {
            "POOR": "poor (<30%)", "LOW": "low (<50%)", "OK": "ok (>50%)"
        },
        "memory_stall_fraction": {
            "HIGH": "high (>0.5)", "ELEVATED": "moderate (0.2-0.5)", "OK": "low (<0.2)"
        },
        "avg_tensor_core_utilization_pct": {
            "POOR": "poor (<20%)", "LOW": "low (<50%)", "OK": "ok (>50%)"
        },
    }
    return notes.get(key, {}).get(status, status.lower())


# ── Bottleneck-specific LLM questions ─────────────────────────────────────────

_BOTTLENECK_QUESTION: dict[str, str] = {
    "inefficient_global_memory_access": (
        "Why does this access pattern cause memory bandwidth saturation? "
        "What is the minimal change to reduce average load sectors per request toward 1.0?"
    ),
    "memory_bandwidth_saturation": (
        "Why is DRAM throughput saturated? "
        "What access pattern or caching change would reduce pressure on memory bandwidth?"
    ),
    "excessive_synchronization": (
        "Which __syncthreads() calls in this kernel are likely unnecessary? "
        "What warp-level primitive (e.g. __syncwarp(), warp shuffle) could safely replace some of them?"
    ),
    "register_pressure": (
        "What technique would reduce register pressure in this kernel without introducing "
        "local-memory spills? Would __launch_bounds__ help here?"
    ),
    "tensor_core_underutilization": (
        "What data type or matrix dimension change would enable tensor core execution for this kernel? "
        "Are there layout or alignment requirements that need to be satisfied?"
    ),
    "warp_divergence_risk": (
        "What conditional branches are causing warp divergence? "
        "How can the kernel be restructured to minimize divergent execution paths?"
    ),
    # NCU bottleneck labels (used when no reconciliation diagnosis)
    "memory_bandwidth_bound": (
        "Why is DRAM throughput saturated? "
        "What access pattern or caching change would reduce memory bandwidth pressure?"
    ),
    "uncoalesced_access": (
        "Why does this access pattern cause uncoalesced loads? "
        "What is the minimal change to ensure consecutive threads access consecutive addresses?"
    ),
    "l1_cache_thrashing": (
        "Why is the L1 cache hit rate so low? "
        "How can the access pattern be restructured to improve cache reuse?"
    ),
    "warp_stall_sync": (
        "Which __syncthreads() calls are likely unnecessary? "
        "What warp-level primitive could replace some of them safely?"
    ),
    "tensor_core_underutilized": (
        "What prevents tensor cores from being used in this kernel? "
        "What data type, layout, or dimension change would enable tensor core execution?"
    ),
    "occupancy_limited_by_registers": (
        "What technique would reduce register pressure without introducing spills? "
        "Would __launch_bounds__ help achieve better occupancy here?"
    ),
    "low_issue_efficiency": (
        "What is causing the warp scheduler to be underutilized? "
        "Would increasing independent work per thread or improving memory access patterns help?"
    ),
}

_DEFAULT_QUESTION = (
    "What is the most impactful optimization for the primary bottleneck identified above? "
    "Please suggest minimal, targeted changes that preserve correctness."
)

# ── Internal helpers ──────────────────────────────────────────────────────────

def _select_primary_diagnosis(diagnoses: list[dict]) -> dict | None:
    if not diagnoses:
        return None
    return sorted(
        diagnoses,
        key=lambda d: (
            _CONFIDENCE_RANK.get(d["confidence"], 99),
            _SEVERITY_RANK.get(d["severity"], 99),
        ),
    )[0]


def _build_missing_data(
    ncu_result: dict | None,
    static_result: dict | None,
) -> list[dict]:
    missing = []
    if static_result is None:
        missing.append({
            "layer": "source",
            "description": "CUDA source not analyzed",
            "how_to_collect": "Re-run with --src kernel.cu to enable static analysis",
        })
    missing.append({
        "layer": "ptx",
        "description": "PTX not analyzed",
        "how_to_collect": "Collect PTX with: nvcc -ptx kernel.cu -o kernel.ptx",
    })
    if ncu_result is not None:
        warnings = ncu_result.get("validation", {}).get("warnings", [])
        if warnings:
            missing.append({
                "layer": "ncu_metrics",
                "description": "NCU metrics may be incomplete",
                "how_to_collect": "Re-run NCU with: ncu --set full --csv report.csv ./your_app",
                "warnings": warnings[:3],
            })
    return missing


def _ncu_evidence_lines(key_metrics: dict) -> list[str]:
    lines = []
    display_order = [
        "avg_dram_throughput_pct",
        "avg_l1_cache_hit_rate_pct",
        "avg_l2_cache_hit_rate_pct",
        "avg_global_load_sectors_per_request",
        "avg_tensor_core_utilization_pct",
        "avg_issue_slot_utilization_pct",
        "avg_occupancy_pct",
        "memory_stall_fraction",
    ]
    for key in display_order:
        val = key_metrics.get(key)
        if val is None:
            continue
        cfg = _METRIC_CONFIG.get(key)
        if cfg is None:
            continue
        label, unit, _, _, _ = cfg
        note = _status_note(key, val)
        if unit:
            lines.append(f"{label}: {val:.1f}{unit} - {note}")
        else:
            lines.append(f"{label}: {val:.2f} - {note}")
    stall = key_metrics.get("dominant_warp_stall")
    stall_pct = key_metrics.get("dominant_warp_stall_pct")
    if stall and stall != "unknown" and stall_pct is not None and stall_pct > 5:
        lines.append(f"Dominant warp stall: {stall} ({stall_pct:.1f}% of cycles)")
    return lines


def _metrics_table_rows(key_metrics: dict) -> list[tuple[str, str, str]]:
    """Returns (label, value_str, status) triples for the LLM prompt table."""
    rows = []
    display_order = [
        "avg_dram_throughput_pct",
        "avg_tensor_core_utilization_pct",
        "avg_l1_cache_hit_rate_pct",
        "avg_l2_cache_hit_rate_pct",
        "avg_global_load_sectors_per_request",
        "avg_issue_slot_utilization_pct",
        "avg_occupancy_pct",
        "memory_stall_fraction",
    ]
    for key in display_order:
        val = key_metrics.get(key)
        if val is None:
            continue
        cfg = _METRIC_CONFIG.get(key)
        if cfg is None:
            continue
        label, unit, _, _, _ = cfg
        status = _metric_status(key, val)
        if unit:
            val_str = f"{val:.1f}{unit}"
        else:
            val_str = f"{val:.3f}" if key == "memory_stall_fraction" else f"{val:.2f}"
        rows.append((label, val_str, status))
    stall = key_metrics.get("dominant_warp_stall")
    stall_pct = key_metrics.get("dominant_warp_stall_pct")
    if stall and stall != "unknown" and stall_pct is not None and stall_pct > 5:
        rows.append(("Dominant Warp Stall", f"{stall} ({stall_pct:.1f}%)", ""))
    return rows


# ── Public API ────────────────────────────────────────────────────────────────

def build_explain_result(
    *,
    ncu_result: dict | None = None,
    static_result: dict | None = None,
    environment: dict | None = None,
) -> dict[str, Any]:
    """Merge available analysis layers into the frx_explain_v0 structure."""
    rec = reconcile_evidence(static=static_result, ncu=ncu_result)

    diagnoses = rec["diagnoses"]
    primary_diag = _select_primary_diagnosis(diagnoses)
    primary_label: str | None = (
        primary_diag["label"] if primary_diag
        else (ncu_result or {}).get("primary_bottleneck")
    )

    ncu_summary = (ncu_result or {}).get("ncu_run_summary", {})
    key_metrics: dict[str, Any] = {}
    for key in _METRIC_CONFIG:
        val = ncu_summary.get(key)
        if val is not None:
            key_metrics[key] = val
    stall = ncu_summary.get("dominant_warp_stall")
    stall_pct = ncu_summary.get("dominant_warp_stall_pct")
    if stall and stall != "unknown":
        key_metrics["dominant_warp_stall"] = stall
    if stall_pct is not None and stall_pct > 0:
        key_metrics["dominant_warp_stall_pct"] = stall_pct

    static_findings = list((static_result or {}).get("findings", []))

    ncu_bottlenecks = [
        {"label": b["label"], "score": b["score"]}
        for b in (ncu_result or {}).get("bottlenecks", [])
    ]

    top_recommendations = [
        {
            "id":                        r["id"],
            "title":                     r["title"],
            "priority":                  r["priority"],
            "tier":                      r["tier"],
            "score":                     r["score"],
            "estimated_speedup_pct_min": r.get("estimated_speedup_pct_min"),
            "estimated_speedup_pct_max": r.get("estimated_speedup_pct_max"),
            "why":                       r.get("why", ""),
            "actions":                   r.get("actions", [])[:2],
            "validation_steps":          r.get("validation_steps", []),
        }
        for r in (ncu_result or {}).get("recommendations", [])[:3]
    ]

    # Top kernel opportunities (ranked by opportunity_score — which kernel to fix first)
    kernel_attr = (ncu_result or {}).get("kernel_attribution", {})
    top_kernels = [
        {
            "kernel_name":     k.get("kernel_name", "unknown"),
            "opportunity":     k.get("opportunity", ""),
            "mfu_pct":         k.get("mfu_pct"),
            "roofline_region": k.get("roofline_region"),
            "runtime_share_pct": k.get("runtime_share_pct"),
        }
        for k in kernel_attr.get("top_opportunities", [])[:5]
    ]

    # Roofline / MFU summary
    roofline = ncu_summary.get("roofline")

    # Occupancy breakdown (only relevant when an occupancy bottleneck is present)
    occupancy_summary = (ncu_result or {}).get("occupancy_summary")

    return {
        "schema": "frx_explain_v0",
        "layers_available": rec["layers_available"],
        "primary_diagnosis": primary_label,
        "diagnoses": diagnoses,
        "key_metrics": key_metrics,
        "static_findings": static_findings,
        "ncu_bottlenecks": ncu_bottlenecks,
        "top_recommendations": top_recommendations,
        "missing_data": _build_missing_data(ncu_result, static_result),
        "top_kernels": top_kernels,
        "roofline": roofline,
        "occupancy_summary": occupancy_summary,
    }


def _speedup_str(rec: dict) -> str:
    """Return '  (est. 10-25% speedup)' when catalog has both bounds, else ''."""
    lo = rec.get("estimated_speedup_pct_min")
    hi = rec.get("estimated_speedup_pct_max")
    if lo is not None and hi is not None:
        return f"  (est. {lo}-{hi}% speedup)"
    return ""


def _render_expected_improvement_lines(recs: list[dict]) -> list[str]:
    """Return lines for the EXPECTED IMPROVEMENT section of the LLM prompt."""
    active = [r for r in recs[:2] if r.get("title")]
    if not active:
        return []

    lines: list[str] = ["**EXPECTED IMPROVEMENT:**", ""]
    for i, r in enumerate(active, 1):
        prefix = f"Fix {i}: " if len(active) > 1 else "Fix: "
        tier   = r.get("tier", "")
        lo     = r.get("estimated_speedup_pct_min")
        hi     = r.get("estimated_speedup_pct_max")
        why    = r.get("why", "").strip()
        vsteps = r.get("validation_steps", [])

        lines.append(f"{prefix}{r['title']}  [{r['priority'].upper()}, Tier: {tier}]")

        if lo is not None and hi is not None:
            lines.append(f"Estimated speedup: {lo}-{hi}%")
        else:
            lines.append("Estimated speedup: not estimated")

        if why:
            for chunk in textwrap.wrap(why, width=72):
                lines.append(chunk)

        if vsteps:
            lines.append("")
            lines.append("Validation targets (re-check these after applying the fix):")
            for step in vsteps:
                direction = step.get("direction", "")
                label     = step.get("label", "")
                expected  = step.get("expected", "")
                threshold = step.get("threshold_good")
                current   = step.get("current_value")

                arrow = {"decrease": "<--", "increase": "-->"}.get(direction, " ~~")
                cur_str = f"was {current:.1f}; " if current is not None else ""
                thr_str = f"  (target: {threshold})" if threshold is not None else ""
                lines.append(f"  {arrow} {label}: {cur_str}{expected}{thr_str}")

            metrics = [s.get("metric", "") for s in vsteps if s.get("metric")]
            if metrics:
                lines.append("")
                lines.append("Re-profile with:")
                lines.append(f"  ncu --metrics {','.join(metrics)} \\")
                lines.append("      --csv after_fix.csv ./your_app")

        if i < len(active):
            lines.append("")

    lines.append("")
    return lines


def render_summary_txt(
    result: dict,
    *,
    ncu_filename: str | None = None,
    src_filename: str | None = None,
) -> str:
    """Human-readable root-cause narrative. Does not duplicate the frx analyze metrics table."""
    lines: list[str] = []
    lines += ["GPU Performance Summary", "=======================", ""]

    if ncu_filename:
        lines.append(f"Source     : {ncu_filename}")
    if src_filename:
        lines.append(f"Kernel src : {src_filename}")
    layers = result.get("layers_available", [])
    lines.append(f"Layers     : {', '.join(layers) if layers else 'ncu only'}")
    lines.append("")

    # Primary issue
    primary_label = result.get("primary_diagnosis")
    diagnoses = result.get("diagnoses", [])
    primary_diag = next((d for d in diagnoses if d["label"] == primary_label), None)

    if primary_diag:
        conf = primary_diag["confidence"]
        sev = primary_diag["severity"]
        confirming = primary_diag["layers_confirming"]
        lines.append(f"PRIMARY ISSUE: {primary_diag['display_name']}")
        lines.append(f"Confidence   : {conf} ({', '.join(confirming)} confirm)")
        lines.append(f"Severity     : {sev}")
    elif primary_label:
        lines.append(f"PRIMARY ISSUE: {primary_label.replace('_', ' ')}")
    else:
        lines.append("PRIMARY ISSUE: (no bottleneck detected above threshold)")
    lines.append("")

    # Evidence
    lines.append("EVIDENCE")
    ev_lines = _ncu_evidence_lines(result.get("key_metrics", {}))
    first = True
    for ev in ev_lines:
        prefix = "  [NCU]    " if first else "           "
        lines.append(f"{prefix}{ev}")
        first = False
    findings = result.get("static_findings", [])
    for f in findings[:3]:
        ln = f" (line {f['line']})" if f.get("line") else ""
        lines.append(f"  [Source] {f['message']}{ln}")
    if not ev_lines and not findings:
        lines.append("  (no metrics available)")
    lines.append("")

    # Root cause
    lines.append("ROOT CAUSE")
    if primary_diag:
        fix = primary_diag.get("fix_summary", "")
        for chunk in textwrap.wrap(fix, width=70):
            lines.append(f"  {chunk}")
    elif primary_label:
        ncu_bns = result.get("ncu_bottlenecks", [])
        top = ncu_bns[0] if ncu_bns else None
        if top:
            lines.append(
                f"  {primary_label.replace('_', ' ').title()} detected "
                f"(score: {top['score']:.2f})."
            )
    else:
        lines.append("  No clear root cause identified from available data.")
    lines.append("")

    # What to fix first
    lines.append("WHAT TO FIX FIRST")
    recs = result.get("top_recommendations", [])
    if recs:
        for i, r in enumerate(recs[:2], 1):
            lines.append(f"  {i}. [{r['priority'].upper()}] {r['title']}{_speedup_str(r)}")
    else:
        lines.append("  (no recommendations generated - check missing data section)")
    lines.append("")

    # Secondary issues
    other_diags = [d for d in diagnoses if d.get("label") != primary_label]
    if other_diags:
        lines.append("SECONDARY ISSUES")
        for d in other_diags[:2]:
            lines.append(f"  - {d['display_name']} ({d['confidence']})")
        lines.append("")

    # Missing data
    missing = result.get("missing_data", [])
    if missing:
        lines.append("MISSING DATA")
        for m in missing:
            lines.append(f"  {m['description']}")
            lines.append(f"  -> {m['how_to_collect']}")
        lines.append("")

    lines.append("Generated by Fournex")
    return "\n".join(lines)


def render_llm_prompt_txt(
    result: dict,
    *,
    kernel_source: str | None = None,
    src_filename: str | None = None,
) -> str:
    """Paste-ready LLM prompt with evidence, guardrails, and a bottleneck-specific question."""
    lines: list[str] = []

    lines += [
        "## CUDA Kernel Optimization Request",
        "",
        "I am analyzing a CUDA kernel for performance optimization.",
        "",
        "**Rules for your response:**",
        "- Do NOT rewrite the entire kernel unless absolutely necessary",
        "- Suggest the minimal targeted change that addresses the identified bottleneck",
        "- Preserve correctness -- flag any correctness risks in your suggestions",
        "- If multiple approaches exist, rank by implementation simplicity",
        "",
        "---",
        "",
    ]

    # Primary bottleneck block
    primary_label = result.get("primary_diagnosis")
    diagnoses = result.get("diagnoses", [])
    primary_diag = next((d for d in diagnoses if d["label"] == primary_label), None)

    if primary_diag:
        conf = primary_diag["confidence"]
        sev = primary_diag["severity"]
        confirming = primary_diag["layers_confirming"]
        lines.append(f"**PRIMARY BOTTLENECK:** {primary_diag['display_name']}")
        lines.append(f"**Confidence:** {conf} ({len(confirming)} layer(s) confirm: {', '.join(confirming)})")
        lines.append(f"**Severity:** {sev}")
    elif primary_label:
        lines.append(f"**PRIMARY BOTTLENECK:** {primary_label.replace('_', ' ').title()}")
    else:
        lines.append("**PRIMARY BOTTLENECK:** No clear bottleneck detected above threshold")
    lines.append("")

    # Secondary bottlenecks
    other_diags = [d for d in diagnoses if d.get("label") != primary_label]
    if other_diags:
        lines.append("**SECONDARY ISSUES (also detected):**")
        for d in other_diags[:3]:
            lines.append(f"- {d['display_name']} [{d['confidence']} confidence]")
        lines.append("")

    # Top kernel opportunities — which kernel to fix first
    top_kernels = result.get("top_kernels", [])
    if len(top_kernels) > 1:
        lines.append("**TOP KERNELS TO OPTIMIZE** (ranked by opportunity):")
        for i, k in enumerate(top_kernels, 1):
            name = k.get("kernel_name", "unknown")
            opp = k.get("opportunity", "")
            mfu = k.get("mfu_pct")
            region = k.get("roofline_region") or ""
            share = k.get("runtime_share_pct")
            parts = []
            if share is not None:
                parts.append(f"runtime: {share:.0f}%")
            if mfu is not None:
                parts.append(f"MFU: {mfu:.0f}%")
            if region:
                parts.append(f"region: {region}")
            if opp:
                parts.append(f"opportunity: {opp}")
            detail = "  " + "  ".join(parts) if parts else ""
            lines.append(f"  {i}. {name}{detail}")
        lines.append("")

    # Expected improvement + validation targets
    lines += _render_expected_improvement_lines(result.get("top_recommendations", []))

    # NCU evidence
    ev_lines = _ncu_evidence_lines(result.get("key_metrics", {}))
    roofline = result.get("roofline")
    occupancy_summary = result.get("occupancy_summary")
    occ_bottleneck = any(
        "occupancy" in (d.get("label") or "") for d in [primary_diag] + other_diags if d
    )
    if ev_lines or roofline or (occ_bottleneck and occupancy_summary):
        lines.append("**EVIDENCE FROM PROFILER (Nsight Compute):**")
        for ev in ev_lines:
            lines.append(f"- {ev}")
        if roofline:
            region = roofline.get("roofline_region", "unknown")
            mfu = roofline.get("mfu_pct")
            mfu_str = f"  MFU: {mfu:.0f}% of peak FP32 throughput" if mfu is not None else ""
            lines.append(f"- Roofline region: {region}{mfu_str}")
        if occ_bottleneck and occupancy_summary:
            limiter = occupancy_summary.get("dominant_limiter")
            eff = occupancy_summary.get("occupancy_efficiency_pct")
            if limiter:
                eff_str = f" (achieved {eff:.0f}% of theoretical)" if eff is not None else ""
                lines.append(f"- Occupancy limiter: {limiter}{eff_str}")
        lines.append("")

    # Static analysis evidence
    findings = result.get("static_findings", [])
    if findings:
        lines.append("**EVIDENCE FROM SOURCE ANALYSIS:**")
        for f in findings[:5]:
            ln = f" (line {f['line']})" if f.get("line") else ""
            lines.append(f"- {f['message']}{ln}")
        lines.append("")

    # Kernel source
    if kernel_source:
        fname = src_filename or "kernel.cu"
        lines += [
            f"**KERNEL SOURCE** (`{fname}`):",
            "```cuda",
            kernel_source.strip(),
            "```",
            "",
        ]
    else:
        lines += [
            "**KERNEL SOURCE:** Not provided.",
            "(Use --src kernel.cu to include source for more targeted advice.)",
            "",
        ]

    # Specific question
    question = _BOTTLENECK_QUESTION.get(primary_label or "", _DEFAULT_QUESTION)
    lines += [
        "**SPECIFIC QUESTION:**",
        question,
        "",
        "---",
        "",
    ]

    # Metrics table
    rows = _metrics_table_rows(result.get("key_metrics", {}))
    if rows:
        col1 = max(len(r[0]) for r in rows)
        col2 = max(len(r[1]) for r in rows)
        col3 = max(len(r[2]) for r in rows) if any(r[2] for r in rows) else 0

        lines.append("**ALL PROFILER METRICS:**")
        header = f"| {'Metric':<{col1}} | {'Value':<{col2}} |"
        if col3:
            header += f" {'Status':<{col3}} |"
        lines.append(header)

        sep = f"| {'-' * col1} | {'-' * col2} |"
        if col3:
            sep += f" {'-' * col3} |"
        lines.append(sep)

        for label, val_str, status in rows:
            row = f"| {label:<{col1}} | {val_str:<{col2}} |"
            if col3:
                row += f" {status:<{col3}} |"
            lines.append(row)
        lines.append("")

    lines += [
        "**NOTE:** After applying the fix, re-profile with the command shown in",
        "EXPECTED IMPROVEMENT above to verify the bottleneck is resolved and no new",
        "bottlenecks were introduced.",
    ]

    return "\n".join(lines)


def render_evidence_json(result: dict) -> str:
    """Serialized JSON of the frx_explain_v0 result (pretty-printed)."""
    return json.dumps(result, indent=2, default=str)


# ── Training-path explain ─────────────────────────────────────────────────────
# Parallel pipeline for PyTorch training telemetry (frx analyze run dirs).
# Produces the same three output files (frx_summary.txt / frx_llm_prompt.txt /
# frx_evidence.json) so users paste the same file into their LLM regardless of
# which profiling path they used.

_TRAINING_BOTTLENECK_QUESTION: dict[str, str] = {
    "input_bound": (
        "My PyTorch training loop spends {dataloader_pct:.0f}% of each step waiting on "
        "the DataLoader. GPU utilization is {gpu_pct:.0f}%. What is the most effective "
        "DataLoader configuration change (num_workers, pin_memory, prefetch_factor, "
        "persistent_workers) or data-preprocessing restructuring to reduce this overhead?"
    ),
    "copy_bound": (
        "My training loop spends {h2d_pct:.0f}% of each step on host-to-device data "
        "transfers. GPU utilization is {gpu_pct:.0f}%. What is the most effective way to "
        "reduce H2D copy overhead — pin_memory, non-blocking transfers, pre-fetching, or "
        "moving preprocessing closer to the GPU?"
    ),
    "sync_bound": (
        "My training loop spends {sync_pct:.0f}% of each step blocked on device "
        "synchronizations. GPU utilization is {gpu_pct:.0f}%. What synchronization calls "
        "are likely unnecessary and what is the safest way to remove or defer them?"
    ),
    "launch_bound": (
        "My training loop launches {kernel_count:.0f} kernels per step with a median "
        "duration of {median_us:.0f}us, leaving GPU utilization at {gpu_pct:.0f}%. "
        "What framework-level optimization — torch.compile, CUDA Graphs, or operator "
        "fusion — would most reduce per-step launch overhead for this workload?"
    ),
    "underutilized_gpu": (
        "GPU utilization is {gpu_pct:.0f}% during training with no dominant data-pipeline "
        "or synchronization stall. What is the most likely cause and what is the most "
        "impactful fix to increase GPU utilization?"
    ),
    "memory_pressure": (
        "Peak GPU memory pressure is {mem_pct:.0f}% of capacity during training. What "
        "technique — gradient checkpointing, mixed precision, micro-batching, or activation "
        "offloading — would most safely reduce peak memory usage without harming throughput?"
    ),
    "shape_instability": (
        "Tensor shapes change in {volatility_pct:.0f}% of training steps, preventing "
        "graph capture and causing per-step recompilation overhead. What is the most "
        "effective way to stabilize shapes for this workload?"
    ),
    "insufficient_telemetry": (
        "Insufficient telemetry was collected to identify a bottleneck. What additional "
        "profiling (SDK instrumentation, torch.profiler, nvidia-smi) would best reveal "
        "the performance bottleneck for this workload?"
    ),
}

_TRAINING_DEFAULT_QUESTION = (
    "What is the most impactful optimization for the primary bottleneck identified above? "
    "Please suggest minimal, targeted changes that preserve training correctness."
)


def _select_scope_data(summary: dict[str, Any], scope: str = "auto") -> dict[str, Any]:
    """Pick the best scope from a summarize_run_with_steady_state result."""
    if scope == "steady_state":
        return summary.get("steady_state") or summary.get("run") or summary
    if scope == "run":
        return summary.get("run") or summary
    # auto: prefer steady_state (skips warmup), fall back to run, then bare dict
    return summary.get("steady_state") or summary.get("run") or summary


def _format_training_question(label: str | None, bottleneck_map: dict, run_summary: dict) -> str:
    template = _TRAINING_BOTTLENECK_QUESTION.get(label or "", _TRAINING_DEFAULT_QUESTION)
    gpu_pct = run_summary.get("average_gpu_utilization_pct", 0.0)
    ev = bottleneck_map.get(label or "", {}).get("evidence", {})
    try:
        return template.format(
            gpu_pct=gpu_pct,
            dataloader_pct=ev.get("avg_dataloader_fraction", 0.0) * 100,
            h2d_pct=ev.get("avg_h2d_fraction", 0.0) * 100,
            sync_pct=ev.get("avg_sync_fraction", 0.0) * 100,
            kernel_count=run_summary.get("kernel_count_per_step", 0.0),
            median_us=run_summary.get("median_cuda_kernel_duration_us", 0.0),
            mem_pct=run_summary.get("memory_pressure_peak_ratio", 0.0) * 100,
            volatility_pct=run_summary.get("shape_volatility_ratio", 0.0) * 100,
        )
    except (KeyError, ValueError):
        return _TRAINING_DEFAULT_QUESTION


def build_telemetry_explain_result(
    *,
    scope_data: dict[str, Any],
) -> dict[str, Any]:
    """Build the frx_telemetry_explain_v0 structure from one scope of training analysis.

    ``scope_data`` is a single scope dict as returned by ``summarize_step_scope()``
    (the value under the "run" or "steady_state" key of ``summarize_run_with_steady_state``).
    """
    diagnosis = scope_data.get("diagnosis", {})
    run_summary = scope_data.get("run_summary", {})
    bottlenecks = scope_data.get("bottlenecks", [])
    fat = scope_data.get("framework_abstraction_tax")

    primary_label: str | None = diagnosis.get("primary_bottleneck")
    user_facing: str | None = diagnosis.get("user_facing_bottleneck", primary_label)
    secondary_labels: list[str] = [
        s for s in diagnosis.get("secondary_bottlenecks", [])
        if s != user_facing and s != primary_label
    ]
    confidence: dict[str, Any] = diagnosis.get("confidence", {})

    bottleneck_map = {b["label"]: b for b in bottlenecks}

    # Key training metrics
    key_metrics: dict[str, Any] = {
        k: run_summary[k]
        for k in (
            "average_gpu_utilization_pct",
            "average_memory_utilization_pct",
            "throughput_steps_per_sec",
            "step_time_avg_ns",
            "memory_pressure_peak_ratio",
            "kernel_count_per_step",
            "median_cuda_kernel_duration_us",
            "small_kernel_fraction",
            "dominant_stall_type",
            "shape_volatility_ratio",
        )
        if run_summary.get(k) is not None
    }

    # Per-step phase fractions from bottleneck evidence
    phase_fractions: dict[str, float] = {}
    for label, ev_key in (
        ("input_bound", "avg_dataloader_fraction"),
        ("copy_bound", "avg_h2d_fraction"),
        ("sync_bound", "avg_sync_fraction"),
    ):
        val = bottleneck_map.get(label, {}).get("evidence", {}).get(ev_key)
        if val is not None:
            phase_fractions[label] = val

    # Top recommendations (from diagnosis, same shape as engine output)
    top_recs = [
        {
            "id":                        r.get("id", ""),
            "title":                     r.get("title", ""),
            "priority":                  r.get("priority", "medium"),
            "tier":                      r.get("tier", ""),
            "score":                     r.get("score", 0.0),
            "estimated_speedup_pct_min": r.get("estimated_speedup_pct_min"),
            "estimated_speedup_pct_max": r.get("estimated_speedup_pct_max"),
            "why":                       r.get("why", ""),
            "actions":                   r.get("actions", [])[:3],
            "validation_steps":          r.get("validation_steps", []),
        }
        for r in diagnosis.get("recommendations", [])[:3]
    ]

    # Missing data hints
    missing_data: list[dict] = []
    if run_summary.get("profiler_windows_exported", 0) == 0:
        missing_data.append({
            "layer": "profiler",
            "description": "No per-step profiler windows captured",
            "how_to_collect": (
                "Instrument your loop with frx.step_context() or wrap with "
                "frx collect -- python train.py"
            ),
        })
    if run_summary.get("kernel_count_per_step") is None:
        missing_data.append({
            "layer": "kernel_trace",
            "description": "No CUDA kernel trace available — launch-overhead signals absent",
            "how_to_collect": "Export a torch.profiler Chrome trace to frx-run/profiler_trace.json",
        })

    return {
        "schema": "frx_telemetry_explain_v0",
        "scope_name": scope_data.get("scope", {}).get("name", "unknown"),
        "step_count": scope_data.get("step_count", 0),
        "primary_bottleneck": user_facing,
        "secondary_bottlenecks": secondary_labels,
        "confidence": confidence,
        "why": diagnosis.get("why", []),
        "key_metrics": key_metrics,
        "phase_fractions": phase_fractions,
        "top_recommendations": top_recs,
        "framework_abstraction_tax": fat,
        "missing_data": missing_data,
        # raw evidence for JSON output
        "bottleneck_evidence": {b["label"]: b.get("evidence", {}) for b in bottlenecks},
    }


def render_training_summary_txt(
    result: dict[str, Any],
    *,
    run_id: str | None = None,
) -> str:
    """Human-readable training bottleneck narrative."""
    lines: list[str] = ["GPU Training Performance Summary", "================================", ""]

    if run_id:
        lines.append(f"Run        : {run_id}")
    scope = result.get("scope_name", "unknown")
    steps = result.get("step_count", 0)
    lines.append(f"Scope      : {scope}  ({steps} steps)")
    lines.append("")

    primary = result.get("primary_bottleneck")
    conf = result.get("confidence", {})
    lines.append(f"PRIMARY BOTTLENECK: {(primary or 'none').replace('_', ' ')}")
    if conf.get("level"):
        lines.append(f"Confidence  : {conf['level']} ({conf.get('score', 0.0):.2f})")
    if conf.get("reason"):
        lines.append(f"Reason      : {conf['reason']}")
    lines.append("")

    why = result.get("why", [])
    if why:
        lines.append("EVIDENCE")
        for bullet in why:
            lines.append(f"  - {bullet}")
        lines.append("")

    km = result.get("key_metrics", {})
    lines.append("KEY METRICS")
    if km.get("average_gpu_utilization_pct") is not None:
        lines.append(f"  GPU Utilization    : {km['average_gpu_utilization_pct']:.1f}%")
    if km.get("throughput_steps_per_sec"):
        lines.append(f"  Throughput         : {km['throughput_steps_per_sec']:,.1f} steps/sec")
    sns = km.get("step_time_avg_ns")
    if sns:
        lines.append(f"  Avg Step Time      : {sns / 1_000_000:.2f} ms")
    if km.get("dominant_stall_type") and km["dominant_stall_type"] != "unknown":
        lines.append(f"  Dominant Stall     : {km['dominant_stall_type']}")
    lines.append("")

    fat = result.get("framework_abstraction_tax")
    if fat and fat.get("score", 0) >= 20:
        lines.append(f"FRAMEWORK ABSTRACTION TAX: {fat['score']}/100 ({fat.get('severity', 'moderate')})")
        for c in fat.get("contributors", []):
            tag = " (inferred)" if c.get("inferred") else ""
            lines.append(f"  - {c['name']}{tag}")
        lines.append("")

    secondary = result.get("secondary_bottlenecks", [])
    if secondary:
        lines.append("SECONDARY ISSUES")
        for s in secondary:
            lines.append(f"  - {s.replace('_', ' ')}")
        lines.append("")

    recs = result.get("top_recommendations", [])
    lines.append("WHAT TO FIX FIRST")
    if recs:
        for i, r in enumerate(recs[:3], 1):
            lines.append(f"  {i}. [{r['priority'].upper()}] {r['title']}{_speedup_str(r)}")
    else:
        lines.append("  (no recommendations generated — check missing data section)")
    lines.append("")

    missing = result.get("missing_data", [])
    if missing:
        lines.append("MISSING DATA")
        for m in missing:
            lines.append(f"  {m['description']}")
            lines.append(f"  -> {m['how_to_collect']}")
        lines.append("")

    lines.append("Generated by Fournex")
    return "\n".join(lines)


def render_training_llm_prompt_txt(result: dict[str, Any]) -> str:
    """Paste-ready LLM prompt for training-loop bottleneck optimization."""
    lines: list[str] = []

    lines += [
        "## PyTorch Training Loop Optimization Request",
        "",
        "I am analyzing a PyTorch training loop for GPU performance optimization.",
        "",
        "**Rules for your response:**",
        "- Suggest the minimal targeted change that addresses the identified bottleneck",
        "- Preserve training correctness - flag any risks (numerical stability, convergence)",
        "- If multiple approaches exist, rank by implementation simplicity",
        "- Do not rewrite the entire training loop unless absolutely necessary",
        "",
        "---",
        "",
    ]

    primary = result.get("primary_bottleneck")
    conf = result.get("confidence", {})
    secondary = [s for s in result.get("secondary_bottlenecks", []) if s != primary]

    if primary:
        lines.append(f"**PRIMARY BOTTLENECK:** {primary.replace('_', ' ')}")
    else:
        lines.append("**PRIMARY BOTTLENECK:** No clear bottleneck detected above threshold")
    if conf.get("level"):
        lines.append(f"**Confidence:** {conf['level']} (score: {conf.get('score', 0.0):.2f})")
    lines.append("")

    if secondary:
        lines.append("**SECONDARY ISSUES (also detected):**")
        for s in secondary:
            lines.append(f"- {s.replace('_', ' ')}")
        lines.append("")

    # Framework Abstraction Tax
    fat = result.get("framework_abstraction_tax")
    if fat and fat.get("score", 0) >= 20:
        lines.append(f"**FRAMEWORK ABSTRACTION TAX: {fat['score']}/100 ({fat.get('severity', 'moderate')})**")
        for c in fat.get("contributors", []):
            tag = " (inferred)" if c.get("inferred") else ""
            lines.append(f"- {c['name']}{tag}")
        lines.append("")

    # Training evidence
    km = result.get("key_metrics", {})
    pf = result.get("phase_fractions", {})
    why = result.get("why", [])

    lines.append("**TRAINING EVIDENCE:**")
    if km.get("average_gpu_utilization_pct") is not None:
        lines.append(f"- GPU utilization: {km['average_gpu_utilization_pct']:.1f}%")
    if km.get("throughput_steps_per_sec"):
        lines.append(f"- Throughput: {km['throughput_steps_per_sec']:,.1f} steps/sec")
    sns = km.get("step_time_avg_ns")
    if sns:
        lines.append(f"- Avg step time: {sns / 1_000_000:.2f} ms")
    if pf.get("input_bound") is not None:
        lines.append(f"- DataLoader fraction: {pf['input_bound'] * 100:.1f}% of step time")
    if pf.get("copy_bound") is not None:
        lines.append(f"- H2D copy fraction: {pf['copy_bound'] * 100:.1f}% of step time")
    if pf.get("sync_bound") is not None:
        lines.append(f"- Sync wait fraction: {pf['sync_bound'] * 100:.1f}% of step time")
    if km.get("kernel_count_per_step") is not None:
        lines.append(f"- Kernel launches/step: {km['kernel_count_per_step']:.0f}")
    if km.get("median_cuda_kernel_duration_us") is not None:
        lines.append(f"- Median kernel duration: {km['median_cuda_kernel_duration_us']:.1f}us")
    if km.get("dominant_stall_type") and km["dominant_stall_type"] != "unknown":
        lines.append(f"- Dominant stall: {km['dominant_stall_type']}")
    if why:
        lines.append("")
        lines.append("Classifier reasoning:")
        for bullet in why[:3]:
            lines.append(f"  {bullet}")
    lines.append("")

    # Top recommendations
    recs = result.get("top_recommendations", [])
    if recs:
        lines.append("**TOP RECOMMENDATIONS:**")
        for i, r in enumerate(recs[:3], 1):
            speedup = _speedup_str(r)
            lines.append(f"")
            lines.append(f"{i}. [{r['priority'].upper()}] {r['title']}{speedup}")
            why_text = r.get("why", "").strip()
            if why_text:
                for chunk in textwrap.wrap(why_text, width=72):
                    lines.append(f"   {chunk}")
            for action in r.get("actions", [])[:3]:
                lines.append(f"   - {action}")
            vsteps = r.get("validation_steps", [])
            if vsteps:
                lines.append("   Validate:")
                for step in vsteps[:2]:
                    direction = step.get("direction", "")
                    label = step.get("label", "")
                    expected = step.get("expected", "")
                    current = step.get("current_value")
                    arrow = {"decrease": "<--", "increase": "-->"}.get(direction, " ~~")
                    cur_str = f"was {current:.1f}; " if current is not None else ""
                    lines.append(f"   {arrow} {label}: {cur_str}{expected}")
        lines.append("")

    # Specific question
    bottleneck_map = {
        label: {"evidence": ev}
        for label, ev in result.get("bottleneck_evidence", {}).items()
    }
    question = _format_training_question(primary, bottleneck_map, km)
    lines += [
        "**SPECIFIC QUESTION:**",
        question,
        "",
        "---",
        "",
        "**NOTE:** After applying the fix, re-collect a training run and compare:",
        "  frx collect --name after-fix -- python train.py",
        "  frx analyze --before <before-run-dir> --after <after-run-dir>",
    ]

    return "\n".join(lines)
