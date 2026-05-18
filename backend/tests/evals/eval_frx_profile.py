"""
Fournex end-to-end accuracy eval: bad_linear_layer.cu

Acts as a user who wrote an inefficient CUDA linear-layer kernel and wants to
know if `frx profile` correctly diagnoses all the problems.

Usage (from WSL):
    cd /mnt/c/Users/jorge/Documents/app_testing2/semiconductor_eda/backend/tests/evals
    python eval_frx_profile.py

Prerequisites: nvcc, ncu, and frx (or fournex) must all be on PATH.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
KERNEL_SRC = EVALS_DIR / "bad_linear_layer.cu"

# ── What Fournex should find ───────────────────────────────────────────────────

EXPECTED_BOTTLENECKS = {
    "uncoalesced_access":        "Flaw 1 — stride-K B_T access, sectors/request >> 4",
    "l1_cache_thrashing":        "Flaw 2 — no shared-memory tiling, large working set",
    "tensor_core_underutilized": "Flaw 3 — FP32 only, tensor cores idle",
    "warp_stall_sync":           "Flaw 4 — spurious __syncthreads() inside K-loop",
}

EXPECTED_REC_IDS = {
    "rec_ncu_improve_coalescing": "fix for Flaw 1",
    "rec_ncu_tiling_shared_mem":  "fix for Flaw 2",
    "rec_ncu_enable_amp":         "fix for Flaw 3",
}

PASS_THRESHOLD_BOTTLENECKS = 3  # must catch at least 3 of 4
PASS_THRESHOLD_RECS = 2         # must suggest at least 2 of 3


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_prerequisites() -> bool:
    missing = [tool for tool in ("nvcc", "ncu", "frx") if not shutil.which(tool)]
    if missing:
        print(f"[SKIP] Prerequisites not on PATH: {', '.join(missing)}")
        print("       Install Nsight Compute and `pip install fournex` in WSL.")
        return False
    return True


def _compile(src: Path) -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        exe = Path(f.name)
    result = subprocess.run(
        ["nvcc", "-O2", "-o", str(exe), str(src)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[ERROR] nvcc failed:\n{result.stderr}")
        sys.exit(1)
    print(f"  Compiled: {src.name}  ->  {exe}")
    return exe


def _profile(exe: Path) -> dict:
    result = subprocess.run(
        ["frx", "profile", "--preset", "full", "--json", "--", str(exe)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[ERROR] frx profile failed (exit {result.returncode}):")
        print(result.stderr[:600])
        sys.exit(1)

    # frx --json wraps output as {"mode": "...", "result": {...}}
    outer = json.loads(result.stdout)
    return outer.get("result", outer)


def _check(label: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return passed


# ── Main eval ──────────────────────────────────────────────────────────────────

def run_eval() -> None:
    print("=" * 60)
    print("  Fournex WSL Eval — bad_linear_layer.cu")
    print("=" * 60)

    if not _check_prerequisites():
        return

    print("\nStep 1: Compile")
    exe = _compile(KERNEL_SRC)

    print("\nStep 2: Profile with frx")
    try:
        profile = _profile(exe)
    finally:
        exe.unlink(missing_ok=True)

    found_bottlenecks = {b["label"]: b for b in profile.get("bottlenecks", [])}
    found_rec_ids     = {r["id"] for r in profile.get("recommendations", [])}
    ncu_summary       = profile.get("ncu_run_summary", {})

    print(f"\n  Kernels profiled : {profile.get('kernel_count', 0)}")
    print(f"  Primary bottleneck: {profile.get('primary_bottleneck', 'none')}")

    # ── Bottleneck checks ──────────────────────────────────────────────────────
    print(f"\nStep 3: Bottleneck detection ({len(EXPECTED_BOTTLENECKS)} expected)")
    bn_passes = 0
    for label, rationale in EXPECTED_BOTTLENECKS.items():
        if label in found_bottlenecks:
            score = found_bottlenecks[label].get("score", 0.0)
            passed = _check(label, True, f"score={score:.2f}")
            bn_passes += 1
        else:
            _check(label, False, rationale)

    # Show key raw metrics to help diagnose any misses
    print(f"\n  Raw NCU metrics:")
    metrics = {
        "avg_dram_throughput_pct":             "DRAM throughput %",
        "avg_l1_cache_hit_rate_pct":           "L1 hit rate %",
        "avg_l2_cache_hit_rate_pct":           "L2 hit rate %",
        "avg_global_load_sectors_per_request": "Sectors/request",
        "avg_tensor_core_utilization_pct":     "Tensor core util %",
        "dominant_warp_stall":                 "Dominant stall",
        "dominant_warp_stall_pct":             "Stall %",
    }
    for key, name in metrics.items():
        val = ncu_summary.get(key)
        if val is not None:
            print(f"    {name:<32} {val}")

    # ── Recommendation checks ──────────────────────────────────────────────────
    print(f"\nStep 4: Recommendation presence ({len(EXPECTED_REC_IDS)} expected)")
    rec_passes = 0
    for rec_id, rationale in EXPECTED_REC_IDS.items():
        if rec_id in found_rec_ids:
            passed = _check(rec_id, True, rationale)
            rec_passes += 1
        else:
            _check(rec_id, False, rationale)

    # ── Final verdict ──────────────────────────────────────────────────────────
    bn_ok  = bn_passes  >= PASS_THRESHOLD_BOTTLENECKS
    rec_ok = rec_passes >= PASS_THRESHOLD_RECS
    overall = "PASS" if (bn_ok and rec_ok) else "FAIL"

    print(f"\n{'=' * 60}")
    print(f"  Result: {overall}")
    print(f"  Bottlenecks: {bn_passes}/{len(EXPECTED_BOTTLENECKS)} detected"
          f"  (need >= {PASS_THRESHOLD_BOTTLENECKS})")
    print(f"  Recommendations: {rec_passes}/{len(EXPECTED_REC_IDS)} present"
          f"  (need >= {PASS_THRESHOLD_RECS})")
    print("=" * 60)

    if overall == "FAIL":
        print("\n  Tip: if most metrics show n/a, ncu may have insufficient")
        print("  hardware counter access (WDDM from WSL2 needs admin privileges).")
        print("  Try: sudo ncu  or run from a native Linux environment.")

    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    run_eval()
