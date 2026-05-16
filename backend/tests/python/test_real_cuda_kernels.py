"""End-to-end PTX analysis tests using real nvcc-compiled CUDA kernels.

Three pairs of kernels in backend/tests/cuda_kernels/ are compiled with
``nvcc -ptx`` and their output is fed through the fournex PTX and comparison
APIs:

  global_memory_heavy.cu  vs  shared_memory_tiled.cu
    -- verifies high_global_memory_ratio + rec_ptx_stage_global_memory

  fp64_compute.cu  vs  fp32_compute.cu
    -- verifies fp64_detected + rec_ptx_reduce_fp64

  register_spills.cu (-maxrregcount 8)  vs  register_bounded.cu
    -- verifies register_spills_detected + rec_ptx_reduce_register_pressure

All tests are skipped gracefully when nvcc is not on PATH.
The NCU integration test is additionally skipped when ncu is absent.

Metric-alias contract (kernel_inspector._canonical_ncu_metric_name):
  Dots in raw metric names become underscores before alias lookup, so
  "dram__throughput.avg.pct_of_peak_sustained_elapsed" is looked up as
  "dram__throughput_avg_pct_of_peak_sustained_elapsed".  The metrics
  requested in the NCU test are chosen to match those exact alias keys.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as at

CUDA_DIR = Path(__file__).resolve().parents[1] / "cuda_kernels"
NVCC = shutil.which("nvcc")
NCU  = shutil.which("ncu")


# ── Compilation helpers ───────────────────────────────────────────────────────

def compile_ptx(cu_path: Path, extra_flags: list[str] | None = None) -> str:
    """Compile *cu_path* to PTX text using nvcc.  Raises pytest.fail on error."""
    extra_flags = extra_flags or []
    with tempfile.NamedTemporaryFile(suffix=".ptx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            ["nvcc", "-ptx", *extra_flags, "-o", str(tmp_path), str(cu_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            pytest.fail(
                f"nvcc failed compiling {cu_path.name}:\n{proc.stderr}"
            )
        return tmp_path.read_text()
    finally:
        tmp_path.unlink(missing_ok=True)


def compile_exe(cu_path: Path, extra_flags: list[str] | None = None) -> Path:
    """Compile *cu_path* to a GPU executable for NCU profiling.

    Returns the path to the binary; caller must unlink it.
    """
    extra_flags = extra_flags or []
    suffix = ".exe" if sys.platform == "win32" else ""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        exe_path = Path(tmp.name)
    proc = subprocess.run(
        ["nvcc", *extra_flags, "-o", str(exe_path), str(cu_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        exe_path.unlink(missing_ok=True)
        pytest.fail(f"nvcc failed compiling {cu_path.name}:\n{proc.stderr}")
    return exe_path


# ── Result helpers ────────────────────────────────────────────────────────────

def _finding_codes(result: dict) -> set[str]:
    return {f["code"] for f in result.get("findings", [])}


def _rec_ids(result: dict) -> set[str]:
    return {r["id"] for r in result.get("recommendations", [])}


# ── Global memory ratio tests ─────────────────────────────────────────────────

@pytest.mark.skipif(not NVCC, reason="nvcc not on PATH")
def test_global_memory_heavy_triggers_stage_global_rec():
    """64 stride-32 global loads exceed the 40% ratio → high_global_memory_ratio + rec."""
    ptx    = compile_ptx(CUDA_DIR / "global_memory_heavy.cu")
    result = at.analyze_ptx_text(ptx)
    codes  = _finding_codes(result)
    recs   = _rec_ids(result)

    ratio = result["run_summary"]["max_global_memory_ratio"]
    assert ratio > 0.40, (
        f"global_memory_heavy expected ratio > 0.40, got {ratio:.3f}. "
        "Inspect the .ptx to check ld.global count vs total instruction count."
    )
    assert "high_global_memory_ratio" in codes, (
        f"Expected high_global_memory_ratio; got codes={codes}"
    )
    assert "rec_ptx_stage_global_memory" in recs, (
        f"Expected rec_ptx_stage_global_memory; got recs={recs}"
    )


@pytest.mark.skipif(not NVCC, reason="nvcc not on PATH")
def test_global_memory_heavy_also_triggers_no_shared_memory_finding():
    """64 global loads with zero shared loads → no_shared_memory_usage finding."""
    ptx    = compile_ptx(CUDA_DIR / "global_memory_heavy.cu")
    result = at.analyze_ptx_text(ptx)
    assert "no_shared_memory_usage" in _finding_codes(result)


@pytest.mark.skipif(not NVCC, reason="nvcc not on PATH")
def test_shared_memory_tiled_has_no_global_memory_finding():
    """Shared-memory tiled kernel: global ratio well below 0.40 → no recommendation."""
    ptx    = compile_ptx(CUDA_DIR / "shared_memory_tiled.cu")
    result = at.analyze_ptx_text(ptx)
    codes  = _finding_codes(result)
    recs   = _rec_ids(result)

    ratio = result["run_summary"]["max_global_memory_ratio"]
    assert ratio < 0.10, (
        f"shared_tiled expected ratio < 0.10, got {ratio:.3f}"
    )
    assert "high_global_memory_ratio" not in codes
    assert "rec_ptx_stage_global_memory" not in recs


# ── FP64 tests ────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not NVCC, reason="nvcc not on PATH")
def test_fp64_kernel_triggers_reduce_fp64_rec():
    """double-precision polynomial → fp64_detected finding + rec_ptx_reduce_fp64."""
    ptx    = compile_ptx(CUDA_DIR / "fp64_compute.cu")
    result = at.analyze_ptx_text(ptx)
    codes  = _finding_codes(result)
    recs   = _rec_ids(result)

    assert result["run_summary"]["has_fp64"], (
        "Expected has_fp64=True; no .f64 arithmetic ops found in PTX"
    )
    assert "fp64_detected" in codes, f"Expected fp64_detected; got codes={codes}"
    assert "rec_ptx_reduce_fp64" in recs, f"Expected rec_ptx_reduce_fp64; got recs={recs}"


@pytest.mark.skipif(not NVCC, reason="nvcc not on PATH")
def test_fp32_kernel_has_no_fp64_finding():
    """float-precision polynomial → no fp64_detected finding."""
    ptx    = compile_ptx(CUDA_DIR / "fp32_compute.cu")
    result = at.analyze_ptx_text(ptx)

    assert not result["run_summary"]["has_fp64"], "Expected has_fp64=False for fp32 kernel"
    assert "fp64_detected" not in _finding_codes(result)
    assert "rec_ptx_reduce_fp64" not in _rec_ids(result)


# ── Register spill tests ──────────────────────────────────────────────────────

@pytest.mark.skipif(not NVCC, reason="nvcc not on PATH")
def test_register_spills_kernel_triggers_register_pressure_rec():
    """volatile local array forces .local PTX memory → register_spills_detected + rec."""
    ptx    = compile_ptx(CUDA_DIR / "register_spills.cu")
    result = at.analyze_ptx_text(ptx)
    codes  = _finding_codes(result)
    recs   = _rec_ids(result)

    assert result["run_summary"]["any_spills"], (
        "Expected .local memory from volatile float arr[16] — architecture-independent spill"
    )
    assert "register_spills_detected" in codes, (
        f"Expected register_spills_detected; got codes={codes}"
    )
    assert "rec_ptx_reduce_register_pressure" in recs, (
        f"Expected rec_ptx_reduce_register_pressure; got recs={recs}"
    )


@pytest.mark.skipif(not NVCC, reason="nvcc not on PATH")
def test_register_bounded_kernel_has_no_spills():
    """Same kernel without -maxrregcount: natural allocation → no spills."""
    ptx    = compile_ptx(CUDA_DIR / "register_bounded.cu")
    result = at.analyze_ptx_text(ptx)

    assert not result["run_summary"]["any_spills"], (
        "Expected no register spills for unconstrained kernel"
    )
    assert "register_spills_detected" not in _finding_codes(result)


# ── compare_implementations tests ─────────────────────────────────────────────

@pytest.mark.skipif(not NVCC, reason="nvcc not on PATH")
def test_global_to_shared_comparison_winner_is_b():
    """Shared-memory version wins memory_efficiency and the overall verdict."""
    ptx_bad  = compile_ptx(CUDA_DIR / "global_memory_heavy.cu")
    ptx_good = compile_ptx(CUDA_DIR / "shared_memory_tiled.cu")

    result = at.compare_implementations(
        {"label": "global_heavy", "ptx": ptx_bad},
        {"label": "shared_tiled", "ptx": ptx_good},
    )
    assert result["verdict"]["overall_winner"] == "b", (
        f"Expected shared_tiled (b) to win; verdict={result['verdict']}, "
        f"scorecard={result['scorecard']}"
    )
    assert result["scorecard"]["memory_efficiency"]["winner"] == "b"


@pytest.mark.skipif(not NVCC, reason="nvcc not on PATH")
def test_global_to_shared_comparison_resolves_global_memory_finding():
    """findings_diff shows high_global_memory_ratio resolved in the shared version."""
    ptx_bad  = compile_ptx(CUDA_DIR / "global_memory_heavy.cu")
    ptx_good = compile_ptx(CUDA_DIR / "shared_memory_tiled.cu")

    result        = at.compare_implementations(
        {"label": "global_heavy", "ptx": ptx_bad},
        {"label": "shared_tiled", "ptx": ptx_good},
    )
    findings_diff = result["ptx_diff"]["findings_diff"]
    assert "high_global_memory_ratio" in findings_diff["resolved_in_b"], (
        f"Expected high_global_memory_ratio in resolved_in_b; diff={findings_diff}"
    )
    assert "high_global_memory_ratio" not in findings_diff["new_in_b"]


@pytest.mark.skipif(not NVCC, reason="nvcc not on PATH")
def test_fp64_to_fp32_comparison_resolves_fp64_finding():
    """findings_diff shows fp64_detected resolved when moving to float precision."""
    ptx_bad  = compile_ptx(CUDA_DIR / "fp64_compute.cu")
    ptx_good = compile_ptx(CUDA_DIR / "fp32_compute.cu")

    result        = at.compare_implementations(
        {"label": "poly_fp64", "ptx": ptx_bad},
        {"label": "poly_fp32", "ptx": ptx_good},
    )
    findings_diff = result["ptx_diff"]["findings_diff"]
    assert "fp64_detected" in findings_diff["resolved_in_b"], (
        f"Expected fp64_detected in resolved_in_b; diff={findings_diff}"
    )
    assert "fp64_detected" not in findings_diff["new_in_b"]
    assert "fp64_detected" not in findings_diff["shared"]


@pytest.mark.skipif(not NVCC, reason="nvcc not on PATH")
def test_spill_to_bounded_comparison_winner_is_b():
    """volatile-array kernel has .local spills; scalar kernel does not → b wins."""
    ptx_bad  = compile_ptx(CUDA_DIR / "register_spills.cu")
    ptx_good = compile_ptx(CUDA_DIR / "register_bounded.cu")

    result = at.compare_implementations(
        {"label": "spill_kernel",   "ptx": ptx_bad},
        {"label": "bounded_kernel", "ptx": ptx_good},
    )
    assert result["verdict"]["overall_winner"] == "b", (
        f"Expected bounded_kernel (b) to win; verdict={result['verdict']}, "
        f"scorecard={result['scorecard']}"
    )
    assert result["scorecard"]["register_efficiency"]["winner"] == "b"


@pytest.mark.skipif(not NVCC, reason="nvcc not on PATH")
def test_spill_to_bounded_comparison_resolves_spill_finding():
    """findings_diff shows register_spills_detected resolved in bounded version."""
    ptx_bad  = compile_ptx(CUDA_DIR / "register_spills.cu")
    ptx_good = compile_ptx(CUDA_DIR / "register_bounded.cu")

    result        = at.compare_implementations(
        {"label": "spill_kernel",   "ptx": ptx_bad},
        {"label": "bounded_kernel", "ptx": ptx_good},
    )
    findings_diff = result["ptx_diff"]["findings_diff"]
    assert "register_spills_detected" in findings_diff["resolved_in_b"], (
        f"Expected register_spills_detected in resolved_in_b; diff={findings_diff}"
    )


# ── NCU integration (requires ncu on PATH and an actual GPU) ──────────────────

@pytest.mark.skipif(not NVCC or not NCU, reason="nvcc or ncu not on PATH")
def test_ncu_profile_of_global_heavy_detects_memory_bottleneck():
    """Compile global_memory_heavy to an executable, profile with ncu, assert memory bottleneck.

    Metric names are chosen to match _canonical_ncu_metric_name() alias keys in
    kernel_inspector.py.  Dots become underscores before lookup, so the raw
    metric name "x.avg.y" must match the alias key "x_avg_y".

    Alias mapping for the metrics collected here:
      dram__throughput.avg.pct_of_peak_sustained_elapsed         → dram_throughput_pct
      sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active → tensor_core_utilization_pct
      l1tex__t_sector_hit_rate.pct                               → l1_cache_hit_rate_pct
      lts__t_sector_hit_rate.pct                                 → l2_cache_hit_rate_pct
      sm__issue_active.avg.pct_of_peak_sustained_active          → issue_slot_utilization_pct
      smsp__pcsamplingdata_pct_of_utilization_issue_stalled_*    → warp_stall_{type}
    """
    # -DBUILD_EXECUTABLE activates the main() block inside global_memory_heavy.cu
    exe = compile_exe(
        CUDA_DIR / "global_memory_heavy.cu",
        extra_flags=["-DBUILD_EXECUTABLE"],
    )
    try:
        ncu_metrics = ",".join([
            # Hardware counter metrics — reliable across all CUDA versions
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
            "l1tex__t_sector_hit_rate.pct",
            "lts__t_sector_hit_rate.pct",
            "sm__issue_active.avg.pct_of_peak_sustained_active",
            # PCsampling stall metrics — feeds warp_stall_breakdown if supported
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_long_scoreboard",
            "smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle",
        ])
        ncu_proc = subprocess.run(
            [
                "ncu", "--csv",
                "--metrics", ncu_metrics,
                "--kernel-name", "global_heavy",
                str(exe),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if ncu_proc.returncode != 0:
            pytest.skip(f"ncu run failed (no GPU or permission?): {ncu_proc.stderr[:300]}")

        ncu_result = at.analyze_ncu_csv_text(ncu_proc.stdout)

        if ncu_result.get("ncu_run_summary", {}).get("kernels_with_ncu_data", 0) == 0:
            pytest.skip(
                "ncu ran but profiled 0 kernels — hardware counters unavailable "
                "(WDDM consumer GPU from WSL2 requires Windows admin access)"
            )

        bottleneck_labels = {b["label"] for b in ncu_result.get("bottlenecks", [])}

        # l1_cache_thrashing fires on L1 hit rate alone (no stall data needed).
        # memory_bandwidth_bound and warp_stall_memory need stall data too.
        # Accept any of the three as valid evidence of memory pressure.
        assert bottleneck_labels & {
            "memory_bandwidth_bound", "warp_stall_memory", "l1_cache_thrashing"
        }, (
            f"Expected a memory bottleneck from ncu profile of global_memory_heavy; "
            f"got {bottleneck_labels}. ncu_run_summary={ncu_result.get('ncu_run_summary')}"
        )
    finally:
        exe.unlink(missing_ok=True)
