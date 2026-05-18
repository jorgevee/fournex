"""Demo: Fournex before/after comparison for a linear-layer CUDA kernel.

Compares bad_linear_layer.cu (naive, flawed) against good_linear_layer.cu
(tiled, coalesced) using static source analysis by default.  Pass --with-ptx
to also compile both kernels with nvcc and feed PTX through the full pipeline.

Run:
    python backend/demos/demo_01_bad_vs_good_linear_layer.py
    python backend/demos/demo_01_bad_vs_good_linear_layer.py --with-ptx
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "python"))

import fournex as at
from fournex.comparison import compare_implementations
from fournex.reconciliation import reconcile_evidence

DEMO_DIR = Path(__file__).parent
BAD_CU   = DEMO_DIR / "bad_linear_layer.cu"
GOOD_CU  = DEMO_DIR / "good_linear_layer.cu"

_SCORE_BAR_WIDTH = 20

# Evidence a PTX or NCU layer would add per finding code.
# "ptx_signal" / "ncu_signal" describe what to look for; used in the table.
_FINDING_EVIDENCE: dict[str, dict] = {
    "unnecessary_syncthreads": {
        "ptx_signal":  "zero shared_load_count in kernel body",
        "ncu_signal":  "warp_stall_sync > 20 %",
        "conf_source": "medium",
        "conf_ptx":    "medium-high",
        "conf_ncu":    "high",
    },
    "missing_obvious_bounds_guard": {
        "ptx_signal":  None,
        "ncu_signal":  None,
        "conf_source": "medium",
        "conf_ptx":    "medium",
        "conf_ncu":    "medium",
    },
    "strided_or_pitched": {
        "ptx_signal":  "global_load_count >> shared_load_count",
        "ncu_signal":  "sectors_per_request > 4.0",
        "conf_source": "medium",
        "conf_ptx":    "medium-high",
        "conf_ncu":    "high",
    },
    "conditional_syncthreads": {
        "ptx_signal":  "barrier inside conditional branch in CFG",
        "ncu_signal":  "warp_stall_sync elevated",
        "conf_source": "high",
        "conf_ptx":    "high",
        "conf_ncu":    "high",
    },
}


# ── Formatting helpers ────────────────────────────────────────────────────────

def _bar(score: float | None) -> str:
    if score is None:
        return "[" + "-" * _SCORE_BAR_WIDTH + "]  n/a"
    filled = round(score * _SCORE_BAR_WIDTH)
    return f"[{'#' * filled}{' ' * (_SCORE_BAR_WIDTH - filled)}]  {score:.2f}"


def _delta_str(delta: float | None) -> str:
    if delta is None:
        return ""
    sign = "+" if delta >= 0 else ""
    return f"  ({sign}{delta:.2f})"


def _col(text: str, width: int) -> str:
    return text.ljust(width)


# ── PTX compilation ───────────────────────────────────────────────────────────

def _compile_ptx(cu_path: Path) -> str | None:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None
    with tempfile.NamedTemporaryFile(suffix=".ptx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [nvcc, "-ptx", "-o", str(tmp_path), str(cu_path)],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            print(f"  [warn] nvcc failed for {cu_path.name}: {proc.stderr[:200]}")
            return None
        return tmp_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"  [warn] nvcc error: {exc}")
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Sections ──────────────────────────────────────────────────────────────────

def _print_findings(result: dict, static_a: dict, static_b: dict, with_ptx: bool) -> None:
    sd = result["static_diff"]
    fd = sd.get("findings_diff", {})
    resolved = fd.get("resolved_in_b", [])
    new_in_b = fd.get("new_in_b", [])
    shared   = fd.get("shared", [])

    findings_a_by_code = {f["code"]: f for f in static_a.get("findings", [])}
    findings_b_by_code = {f["code"]: f for f in static_b.get("findings", [])}

    print()
    print("FINDINGS RESOLVED IN B")
    print("-" * 62)
    if resolved:
        for code in resolved:
            f = findings_a_by_code.get(code, {})
            sev = f.get("severity", "?")
            sym = {"high": "!!", "medium": "! ", "low": "  "}.get(sev, "  ")
            print(f"  {sym} [{sev}] {code}")
            if f.get("message"):
                print(f"       {f['message']}")
    else:
        print("  (none)")

    print()
    print("NEW FINDINGS IN B")
    print("-" * 62)
    if new_in_b:
        for code in new_in_b:
            f = findings_b_by_code.get(code, {})
            sev = f.get("severity", "?")
            sym = {"high": "!!", "medium": "! ", "low": "  "}.get(sev, "  ")
            print(f"  {sym} [{sev}] {code}")
    else:
        print("  (none - no regressions introduced)")

    if shared:
        print()
        print("FINDINGS PRESENT IN BOTH")
        print("-" * 62)
        for code in shared:
            print(f"       {code}")


def _print_evidence_table(result: dict, static_a: dict, with_ptx: bool) -> None:
    sd = result["static_diff"]
    fd = sd.get("findings_diff", {})
    resolved = fd.get("resolved_in_b", [])

    mem_added   = sd.get("memory_access_styles", {}).get("added_in_b", [])
    mem_removed = sd.get("memory_access_styles", {}).get("removed_in_b", [])

    rows: list[tuple[str, str, str, str, str]] = []  # finding, source, ptx, ncu, confidence

    ptx_col = "yes" if with_ptx else "n/a"

    for code in resolved:
        ev = _FINDING_EVIDENCE.get(code, {})
        if with_ptx and ev.get("ptx_signal"):
            ptx_status = "yes"
            conf = ev.get("conf_ptx", "medium")
        else:
            ptx_status = ptx_col
            conf = ev.get("conf_source", "medium")
        rows.append((code, "yes", ptx_status, "n/a", conf))

    for style in mem_removed:
        ev = _FINDING_EVIDENCE.get(style, {})
        rows.append((f"{style} (removed)", "yes", ptx_col, "n/a",
                     ev.get("conf_source", "medium")))

    for style in mem_added:
        rows.append((f"{style} (added)", "yes", ptx_col, "n/a", "high"))

    if not rows:
        return

    print()
    print("EVIDENCE TABLE")
    print("-" * 62)
    w_finding = max(len(r[0]) for r in rows) + 2
    header = (
        _col("Finding", w_finding)
        + _col("Source", 8)
        + _col("PTX", 6)
        + _col("NCU", 6)
        + "Confidence"
    )
    print("  " + header)
    print("  " + "-" * (len(header) + 2))
    for finding, src, ptx, ncu, conf in rows:
        print("  " + _col(finding, w_finding) + _col(src, 8) + _col(ptx, 6) + _col(ncu, 6) + conf)

    if not with_ptx:
        print()
        ptx_upgrades = [
            (code, _FINDING_EVIDENCE[code]["ptx_signal"])
            for code in resolved
            if code in _FINDING_EVIDENCE and _FINDING_EVIDENCE[code].get("ptx_signal")
        ]
        if ptx_upgrades:
            print("  PTX would upgrade confidence for:")
            for code, signal in ptx_upgrades:
                print(f"    {code}: {signal}")


def _print_structural(result: dict) -> None:
    sd = result["static_diff"]
    smem = sd.get("shared_memory_alloc_count", {})
    smem_a, smem_b = smem.get("a", 0), smem.get("b", 0)

    changes: list[str] = []
    if smem_a != smem_b:
        direction = "added" if smem_b > smem_a else "removed"
        changes.append(f"  shared memory tiles: {smem_a} -> {smem_b}  ({direction})")

    for style in sd.get("memory_access_styles", {}).get("added_in_b", []):
        changes.append(f"  + access style added:   {style}")
    for style in sd.get("memory_access_styles", {}).get("removed_in_b", []):
        changes.append(f"  - access style removed: {style}")

    bank = sd.get("has_bank_conflict_risk", {})
    if bank.get("resolved_in_b"):
        changes.append("  + bank conflict risk: resolved")
    elif bank.get("introduced_in_b"):
        changes.append("  - bank conflict risk: introduced")

    if changes:
        print()
        print("STRUCTURAL CHANGES")
        print("-" * 62)
        for line in changes:
            print(line)


def _print_scorecard(result: dict, with_ptx: bool) -> None:
    sc = result["scorecard"]
    dim_labels = {
        "launch_efficiency":   "launch efficiency  ",
        "sync_efficiency":     "sync efficiency    ",
        "memory_efficiency":   "memory efficiency  ",
        "compute_efficiency":  "compute efficiency ",
        "register_efficiency": "register efficiency",
    }
    print()
    print("SCORECARD")
    print("-" * 62)
    for dim, label in dim_labels.items():
        d = sc.get(dim, {})
        sa, sb = d.get("score_a"), d.get("score_b")
        w = d.get("weight", 0)
        if not d.get("available"):
            hint = "PTX or NCU" if dim in ("memory_efficiency", "compute_efficiency", "register_efficiency") else "source"
            print(f"  {label}  (unavailable - {hint} required)")
            continue
        delta = (sb - sa) if sa is not None and sb is not None else None
        winner_tag = "  <-- winner" if d.get("winner") == "b" else ("  <-- A wins" if d.get("winner") == "a" else "")
        print(f"  {label}  A: {_bar(sa)}")
        print(f"  {' ' * len(label)}  B: {_bar(sb)}{_delta_str(delta)}{winner_tag}")


def _print_reconciliation(static_a: dict, static_b: dict, ptx_bad: str | None, ptx_good: str | None) -> None:
    """Show cross-layer reconciliation for the bad kernel (A side only)."""
    from fournex.ptx_analysis import analyze_ptx_text

    ptx_result = analyze_ptx_text(ptx_bad) if ptx_bad else None
    rec = reconcile_evidence(static=static_a, ptx=ptx_result)

    print()
    print("CROSS-LAYER RECONCILIATION (A: bad_linear_layer.cu)")
    print("-" * 62)
    if not rec["diagnoses"]:
        print("  (no multi-layer diagnoses with current evidence)")
        return

    layers_str = " + ".join(rec["layers_available"])
    print(f"  Evidence layers: {layers_str}")
    print()
    for d in rec["diagnoses"]:
        sev_sym = {"high": "!!", "medium": "! ", "low": "  "}.get(d["severity"], "  ")
        confirming_str = ", ".join(d["layers_confirming"])
        print(f"  {sev_sym} {d['display_name']}")
        print(f"     Confidence : {d['confidence']}  (confirmed by: {confirming_str})")
        print(f"     Fix        : {d['fix_summary']}")
    if rec["unreconciled"]:
        print()
        print("  Unreconciled findings (no matching diagnosis):")
        for layer, codes in rec["unreconciled"].items():
            print(f"    [{layer}] {', '.join(codes)}")


def _print_verdict(result: dict, label_a: str, label_b: str) -> None:
    vd = result["verdict"]
    winner = vd.get("overall_winner")
    sa_v, sb_v = vd.get("score_a"), vd.get("score_b")
    delta_v = vd.get("score_delta")

    winner_label = label_b if winner == "b" else (label_a if winner == "a" else "tie")

    print()
    print("VERDICT")
    print("-" * 62)
    print(f"  Winner:   {winner_label}")
    if sa_v is not None:
        print(f"  Score A:  {sa_v:.3f}")
    if sb_v is not None:
        print(f"  Score B:  {sb_v:.3f}")
    if delta_v is not None:
        sign = "+" if delta_v >= 0 else ""
        print(f"  Delta:    {sign}{delta_v:.3f}")
    dims_b = vd.get("dimensions_won_by_b", [])
    if dims_b:
        print(f"  B won:    {', '.join(dims_b)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fournex linear-layer before/after demo")
    parser.add_argument("--with-ptx", action="store_true",
                        help="Compile both kernels with nvcc and include PTX analysis")
    args = parser.parse_args()

    bad_src  = BAD_CU.read_text(encoding="utf-8")
    good_src = GOOD_CU.read_text(encoding="utf-8")

    ptx_bad = ptx_good = None
    if args.with_ptx:
        nvcc = shutil.which("nvcc")
        if not nvcc:
            print("[warn] --with-ptx requested but nvcc not found on PATH; falling back to source-only")
        else:
            print("[info] Compiling kernels with nvcc ...")
            ptx_bad  = _compile_ptx(BAD_CU)
            ptx_good = _compile_ptx(GOOD_CU)

    layer_desc = "CUDA source"
    if ptx_bad and ptx_good:
        layer_desc = "CUDA source + PTX"

    result = compare_implementations(
        {"label": "bad_linear_layer.cu",  "cuda_source": bad_src,  "cuda_filename": "bad_linear_layer.cu",
         "ptx": ptx_bad,  "ptx_filename": "bad_linear_layer.ptx"},
        {"label": "good_linear_layer.cu", "cuda_source": good_src, "cuda_filename": "good_linear_layer.cu",
         "ptx": ptx_good, "ptx_filename": "good_linear_layer.ptx"},
    )

    label_a = result["label_a"]
    label_b = result["label_b"]
    static_a = at.inspect_cuda_source(bad_src,  filename="bad_linear_layer.cu")
    static_b = at.inspect_cuda_source(good_src, filename="good_linear_layer.cu")

    print()
    print("=" * 62)
    print("  Fournex before/after comparison")
    print(f"  A: {label_a}")
    print(f"  B: {label_b}")
    print(f"  Analysis: {layer_desc}")
    print("=" * 62)

    _print_findings(result, static_a, static_b, args.with_ptx)
    _print_evidence_table(result, static_a, args.with_ptx)
    _print_structural(result)
    _print_scorecard(result, args.with_ptx)
    _print_verdict(result, label_a, label_b)
    _print_reconciliation(static_a, static_b, ptx_bad, ptx_good)

    print()
    if not (ptx_bad and ptx_good):
        print("  To unlock memory/compute/register efficiency scores:")
        print("    python backend/demos/demo_01_bad_vs_good_linear_layer.py --with-ptx")
        print("  (requires nvcc on PATH)")
    else:
        print("  To unlock hardware bottleneck scores, run ncu --set full on both kernels")
        print("  and pass the CSV outputs via analyze_ncu_csv_text().")
    print()
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
