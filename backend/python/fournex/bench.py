from __future__ import annotations

import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .ncu_comparison import diff_ncu_runs

# Sentinel printed by frx_bench_harness.cuh: profiler-free per-launch GPU time.
_KERNEL_US_RE = re.compile(r"FRX_KERNEL_US:\s*([0-9]+(?:\.[0-9]+)?)")


def harness_include_dir() -> Path:
    """Directory containing frx_bench_harness.cuh, shipped as package data."""
    return Path(__file__).resolve().parent / "data"


def harness_header_path() -> Path:
    return harness_include_dir() / "frx_bench_harness.cuh"


def _parse_kernel_event_us(stdout: str) -> float | None:
    """Extract the FRX_KERNEL_US sentinel (cudaEvent timing) from binary stdout."""
    match = _KERNEL_US_RE.search(stdout or "")
    return float(match.group(1)) if match else None


def compile_kernel(
    src_path: Path,
    out_path: Path,
    *,
    build_flags: str = "-DBUILD_EXECUTABLE",
    arch: str | None = None,
) -> tuple[bool, str]:
    """Compile src_path with nvcc. Returns (success, stderr_text)."""
    cmd = ["nvcc", str(src_path), "-o", str(out_path)]
    if arch:
        cmd += ["-arch", arch]
    # Put the bench harness on the include path so any kernel can opt into
    # cudaEvent timing with a bare `#include "frx_bench_harness.cuh"`.
    harness_dir = harness_include_dir()
    if harness_dir.is_dir():
        cmd += ["-I", str(harness_dir)]
    for flag in build_flags.split():
        if flag:
            cmd.append(flag)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.returncode == 0, result.stderr


def time_binary(
    exe_path: Path,
    *,
    warmup: int = 2,
    runs: int = 5,
) -> dict[str, Any]:
    """Wall-clock time exe_path. Warmup runs are discarded. Returns timing stats in ms.

    If the binary prints the FRX_KERNEL_US sentinel (from frx_bench_harness.cuh),
    its median across timed runs is reported as ``kernel_event_us`` — a
    profiler-free, init-free measure of the kernel region itself.
    """
    all_ms: list[float] = []
    event_us: list[float] = []
    for i in range(warmup + runs):
        t0 = time.perf_counter()
        proc = subprocess.run(
            [str(exe_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if proc.returncode != 0:
            raise RuntimeError(
                f"Binary exited with code {proc.returncode}: {proc.stderr[:200]}"
            )
        if i >= warmup:
            all_ms.append(elapsed_ms)
            ku = _parse_kernel_event_us(proc.stdout)
            if ku is not None:
                event_us.append(ku)
    return {
        "median_ms": round(statistics.median(all_ms), 3),
        "min_ms":    round(min(all_ms), 3),
        "max_ms":    round(max(all_ms), 3),
        "stdev_ms":  round(statistics.stdev(all_ms) if len(all_ms) > 1 else 0.0, 3),
        "kernel_event_us": round(statistics.median(event_us), 4) if event_us else None,
        "runs":      runs,
        "warmup":    warmup,
    }


def profile_with_ncu(
    exe_path: Path,
    *,
    preset: str = "full",
    sm_version: str | int | None = None,
) -> str | None:
    """Run NCU on exe_path. Return filtered CSV text or None on NCU failure."""
    from .ncu_presets import build_ncu_command
    cmd = build_ncu_command(preset, [str(exe_path)], sm_version=sm_version)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        return None
    # Keep only quoted CSV lines; drops "done", app stdout, ==PROF== lines
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith('"')]
    return "\n".join(lines) if lines else None


def bench_compare(
    before_src: Path,
    after_src: Path,
    *,
    warmup: int = 2,
    runs: int = 5,
    with_ncu: bool = False,
    arch: str | None = None,
    build_flags: str = "-DBUILD_EXECUTABLE",
    out_dir: Path | None = None,
    preset: str = "full",
) -> dict[str, Any]:
    """Compile, time, and optionally profile two .cu files. Return frx_bench_v0 result dict."""
    compile_errors: list[dict[str, str]] = []

    _cleanup = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir = out_dir
    else:
        _tmp = tempfile.TemporaryDirectory()
        work_dir = Path(_tmp.name)
        _cleanup = _tmp

    try:
        exe_suffix = ".exe" if sys.platform == "win32" else ""
        before_exe = work_dir / f"frx_bench_before{exe_suffix}"
        after_exe  = work_dir / f"frx_bench_after{exe_suffix}"

        t0 = time.perf_counter()
        ok_b, err_b = compile_kernel(before_src, before_exe, build_flags=build_flags, arch=arch)
        compile_before_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        if not ok_b:
            compile_errors.append({"side": "before", "src": str(before_src), "error": err_b})

        t0 = time.perf_counter()
        ok_a, err_a = compile_kernel(after_src, after_exe, build_flags=build_flags, arch=arch)
        compile_after_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        if not ok_a:
            compile_errors.append({"side": "after", "src": str(after_src), "error": err_a})

        if compile_errors:
            return {
                "schema":         "frx_bench_v0",
                "arch":           arch,
                "before":         {"src": str(before_src), "compile_ms": compile_before_ms, "timing": None},
                "after":          {"src": str(after_src),  "compile_ms": compile_after_ms,  "timing": None},
                "speedup_x":      None,
                "kernel_event":   {"available": False, "baseline_us": None, "optimized_us": None, "speedup_x": None},
                "event_speedup_x": None,
                "kernel_speedup_x": None,
                "primary_speedup_x": None,
                "primary_speedup_basis": None,
                "ncu_diff":       None,
                "compile_errors": compile_errors,
            }

        before_timing = time_binary(before_exe, warmup=warmup, runs=runs)
        after_timing  = time_binary(after_exe,  warmup=warmup, runs=runs)

        after_median = after_timing["median_ms"]
        speedup_x = round(before_timing["median_ms"] / after_median, 3) if after_median > 0 else None

        ncu_diff = None
        if with_ncu:
            # arch (e.g. "sm_120") doubles as the sm hint so PC-sampling metrics
            # are dropped on Blackwell instead of failing the whole ncu pass.
            before_csv = profile_with_ncu(before_exe, preset=preset, sm_version=arch)
            after_csv  = profile_with_ncu(after_exe,  preset=preset, sm_version=arch)
            if before_csv and after_csv:
                ncu_diff = diff_ncu_runs(
                    before_csv,
                    after_csv,
                    label_baseline=before_src.name,
                    label_optimized=after_src.name,
                )

        # cudaEvent timing (from frx_bench_harness.cuh): profiler-free, init-free
        # measure of the kernel region. The best basis when present.
        before_event_us = before_timing.get("kernel_event_us")
        after_event_us  = after_timing.get("kernel_event_us")
        event_available = (
            before_event_us is not None and after_event_us is not None and after_event_us > 0
        )
        event_speedup_x = round(before_event_us / after_event_us, 4) if event_available else None
        kernel_event = {
            "available":    event_available,
            "baseline_us":  before_event_us,
            "optimized_us": after_event_us,
            "speedup_x":    event_speedup_x,
            "source":       "cudaEvent (in-binary harness, profiler-free)",
        }

        # NCU kernel GPU time: also kernel-only, but profiler-serialized (absolute
        # inflated, ratio valid). Used when the cudaEvent harness isn't present.
        kernel_speedup_x = None
        if ncu_diff is not None:
            kt = ncu_diff.get("kernel_time") or {}
            if kt.get("available"):
                kernel_speedup_x = kt.get("speedup_x")

        # Basis priority: profiler-free cudaEvent > NCU duration > whole-process wall.
        if event_speedup_x is not None:
            primary_speedup_x = event_speedup_x
            primary_speedup_basis = "cuda_event"
        elif kernel_speedup_x is not None:
            primary_speedup_x = kernel_speedup_x
            primary_speedup_basis = "kernel_gpu_time"
        else:
            primary_speedup_x = speedup_x
            primary_speedup_basis = "wall_clock"

        return {
            "schema":         "frx_bench_v0",
            "arch":           arch,
            "before":         {"src": str(before_src), "compile_ms": compile_before_ms, "timing": before_timing},
            "after":          {"src": str(after_src),  "compile_ms": compile_after_ms,  "timing": after_timing},
            "speedup_x":      speedup_x,
            "kernel_event":   kernel_event,
            "event_speedup_x": event_speedup_x,
            "kernel_speedup_x": kernel_speedup_x,
            "primary_speedup_x": primary_speedup_x,
            "primary_speedup_basis": primary_speedup_basis,
            "ncu_diff":       ncu_diff,
            "compile_errors": compile_errors,
        }
    finally:
        if _cleanup is not None:
            _cleanup.cleanup()
