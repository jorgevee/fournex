"""Tests for frx bench — compile-and-time two .cu kernels."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex.bench import (
    bench_compare,
    compile_kernel,
    profile_with_ncu,
    time_binary,
)


# ── compile_kernel ────────────────────────────────────────────────────────────

def test_compile_kernel_success():
    with patch("fournex.bench.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        ok, err = compile_kernel(Path("kernel.cu"), Path("kernel.exe"))
    assert ok is True
    assert err == ""


def test_compile_kernel_failure():
    with patch("fournex.bench.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="error: undeclared identifier")
        ok, err = compile_kernel(Path("kernel.cu"), Path("kernel.exe"))
    assert ok is False
    assert "undeclared identifier" in err


def test_compile_kernel_includes_arch_flag():
    with patch("fournex.bench.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        compile_kernel(Path("k.cu"), Path("k.exe"), arch="sm_120")
        cmd = mock_run.call_args[0][0]
    assert "-arch" in cmd
    assert "sm_120" in cmd


def test_compile_kernel_includes_build_flags():
    with patch("fournex.bench.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        compile_kernel(Path("k.cu"), Path("k.exe"), build_flags="-DBUILD_EXECUTABLE -O3")
        cmd = mock_run.call_args[0][0]
    assert "-DBUILD_EXECUTABLE" in cmd
    assert "-O3" in cmd


def test_compile_kernel_no_arch_by_default():
    with patch("fournex.bench.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        compile_kernel(Path("k.cu"), Path("k.exe"))
        cmd = mock_run.call_args[0][0]
    assert "-arch" not in cmd


def test_compile_kernel_nvcc_is_first_in_cmd():
    with patch("fournex.bench.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        compile_kernel(Path("k.cu"), Path("k.exe"))
        cmd = mock_run.call_args[0][0]
    assert cmd[0] == "nvcc"


# ── time_binary ───────────────────────────────────────────────────────────────

def _perf_counter_sequence(*pairs_ms: float):
    """Return a side_effect list: pairs of (0.0, elapsed_s) for each run_ms."""
    effects = []
    for ms in pairs_ms:
        effects.append(0.0)
        effects.append(ms / 1000.0)
    return effects


def test_time_binary_warmup_discarded():
    # warmup=2 at 200ms each, runs=3 at 100ms each
    times = _perf_counter_sequence(200.0, 200.0, 100.0, 100.0, 100.0)
    with patch("fournex.bench.subprocess.run") as mock_run, \
         patch("fournex.bench.time.perf_counter", side_effect=times):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        result = time_binary(Path("fake.exe"), warmup=2, runs=3)
    assert result["runs"] == 3
    assert result["warmup"] == 2
    assert result["median_ms"] == pytest.approx(100.0, abs=0.5)


def test_time_binary_stdev_zero_for_identical_runs():
    times = _perf_counter_sequence(50.0, 50.0, 50.0, 50.0, 50.0)
    with patch("fournex.bench.subprocess.run") as mock_run, \
         patch("fournex.bench.time.perf_counter", side_effect=times):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        result = time_binary(Path("fake.exe"), warmup=0, runs=5)
    assert result["stdev_ms"] == 0.0


def test_time_binary_stats_correct():
    # 5 runs: 10, 20, 30, 40, 50 ms — median=30, min=10, max=50
    times = _perf_counter_sequence(10.0, 20.0, 30.0, 40.0, 50.0)
    with patch("fournex.bench.subprocess.run") as mock_run, \
         patch("fournex.bench.time.perf_counter", side_effect=times):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        result = time_binary(Path("fake.exe"), warmup=0, runs=5)
    assert result["median_ms"] == pytest.approx(30.0, abs=0.5)
    assert result["min_ms"] == pytest.approx(10.0, abs=0.5)
    assert result["max_ms"] == pytest.approx(50.0, abs=0.5)


def test_time_binary_raises_on_nonzero_exit():
    times = _perf_counter_sequence(10.0)
    with patch("fournex.bench.subprocess.run") as mock_run, \
         patch("fournex.bench.time.perf_counter", side_effect=times):
        mock_run.return_value = MagicMock(returncode=1, stderr="segfault", stdout="")
        with pytest.raises(RuntimeError, match="exited with code 1"):
            time_binary(Path("fake.exe"), warmup=0, runs=1)


def test_time_binary_result_keys():
    times = _perf_counter_sequence(10.0, 10.0, 10.0)
    with patch("fournex.bench.subprocess.run") as mock_run, \
         patch("fournex.bench.time.perf_counter", side_effect=times):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        result = time_binary(Path("fake.exe"), warmup=0, runs=3)
    for key in ("median_ms", "min_ms", "max_ms", "stdev_ms", "runs", "warmup"):
        assert key in result


# ── profile_with_ncu ─────────────────────────────────────────────────────────

_FAKE_NCU_STDOUT = (
    '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
    '"myKernel","dram__throughput.avg.pct_of_peak_sustained_elapsed","percent","82.5"\n'
    "done\n"
    "Application stdout line\n"
    "==PROF== profiler warning\n"
)


def test_profile_with_ncu_filters_non_quoted_lines():
    with patch("fournex.bench.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=_FAKE_NCU_STDOUT, stderr="")
        csv_text = profile_with_ncu(Path("kernel.exe"))
    assert csv_text is not None
    assert "done" not in csv_text
    assert "Application stdout" not in csv_text
    assert "==PROF==" not in csv_text
    assert '"Kernel Name"' in csv_text


def test_profile_with_ncu_returns_none_on_failure():
    with patch("fournex.bench.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="access denied")
        result = profile_with_ncu(Path("kernel.exe"))
    assert result is None


def test_profile_with_ncu_returns_none_when_no_csv_rows():
    with patch("fournex.bench.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="done\n==PROF==\n", stderr="")
        result = profile_with_ncu(Path("kernel.exe"))
    assert result is None


# ── bench_compare ─────────────────────────────────────────────────────────────

def _timing(median: float, runs: int = 5, warmup: int = 2) -> dict:
    return {
        "median_ms": median,
        "min_ms":    median,
        "max_ms":    median,
        "stdev_ms":  0.0,
        "runs":      runs,
        "warmup":    warmup,
    }


def test_bench_compare_schema():
    with patch("fournex.bench.compile_kernel", return_value=(True, "")), \
         patch("fournex.bench.time_binary", side_effect=[_timing(10.0), _timing(8.0)]):
        result = bench_compare(Path("bad.cu"), Path("good.cu"))
    assert result["schema"] == "frx_bench_v0"


def test_bench_compare_required_fields():
    with patch("fournex.bench.compile_kernel", return_value=(True, "")), \
         patch("fournex.bench.time_binary", side_effect=[_timing(10.0), _timing(8.0)]):
        result = bench_compare(Path("bad.cu"), Path("good.cu"))
    for field in ("schema", "arch", "before", "after", "speedup_x", "ncu_diff", "compile_errors"):
        assert field in result, f"missing: {field}"


def test_bench_compare_speedup_calculated():
    with patch("fournex.bench.compile_kernel", return_value=(True, "")), \
         patch("fournex.bench.time_binary", side_effect=[_timing(10.0), _timing(8.0)]):
        result = bench_compare(Path("bad.cu"), Path("good.cu"))
    assert result["speedup_x"] == pytest.approx(10.0 / 8.0, rel=0.01)


def test_bench_compare_speedup_gt_one_means_after_is_faster():
    with patch("fournex.bench.compile_kernel", return_value=(True, "")), \
         patch("fournex.bench.time_binary", side_effect=[_timing(20.0), _timing(10.0)]):
        result = bench_compare(Path("bad.cu"), Path("good.cu"))
    assert result["speedup_x"] == pytest.approx(2.0, rel=0.01)


def test_bench_compare_ncu_diff_none_without_flag():
    with patch("fournex.bench.compile_kernel", return_value=(True, "")), \
         patch("fournex.bench.time_binary", side_effect=[_timing(10.0), _timing(8.0)]):
        result = bench_compare(Path("bad.cu"), Path("good.cu"), with_ncu=False)
    assert result["ncu_diff"] is None


def test_bench_compare_arch_stored_in_result():
    with patch("fournex.bench.compile_kernel", return_value=(True, "")), \
         patch("fournex.bench.time_binary", side_effect=[_timing(10.0), _timing(8.0)]):
        result = bench_compare(Path("bad.cu"), Path("good.cu"), arch="sm_120")
    assert result["arch"] == "sm_120"


def test_bench_compare_arch_none_by_default():
    with patch("fournex.bench.compile_kernel", return_value=(True, "")), \
         patch("fournex.bench.time_binary", side_effect=[_timing(10.0), _timing(8.0)]):
        result = bench_compare(Path("bad.cu"), Path("good.cu"))
    assert result["arch"] is None


def test_bench_compare_compile_error_propagates():
    err_text = "error: 'foo' undeclared"
    with patch("fournex.bench.compile_kernel", return_value=(False, err_text)):
        result = bench_compare(Path("bad.cu"), Path("good.cu"))
    assert len(result["compile_errors"]) >= 1
    assert any(e["error"] == err_text for e in result["compile_errors"])
    assert result["speedup_x"] is None
    assert result["before"]["timing"] is None
    assert result["after"]["timing"] is None


def test_bench_compare_compile_error_skips_timing():
    with patch("fournex.bench.compile_kernel", return_value=(False, "bad")), \
         patch("fournex.bench.time_binary") as mock_time:
        bench_compare(Path("bad.cu"), Path("good.cu"))
    mock_time.assert_not_called()


def test_bench_compare_with_ncu_calls_profile_twice():
    fake_csv = '"Kernel Name","Metric Name"\n"k","v"\n'
    fake_diff = {
        "schema": "ncu_comparison_v1",
        "bottleneck_diff": {"resolved": [], "new": [], "persistent": [], "improved": [], "score_deltas": {}},
        "metric_deltas": {},
        "verdict": {"outcome": "neutral", "bottlenecks_resolved": 0, "bottlenecks_new": 0, "bottlenecks_persistent": 0, "bottlenecks_improved": 0},
        "baseline":  {"bottlenecks": []},
        "optimized": {"bottlenecks": []},
    }
    with patch("fournex.bench.compile_kernel", return_value=(True, "")), \
         patch("fournex.bench.time_binary", side_effect=[_timing(10.0), _timing(8.0)]), \
         patch("fournex.bench.profile_with_ncu", return_value=fake_csv) as mock_profile, \
         patch("fournex.bench.diff_ncu_runs", return_value=fake_diff):
        result = bench_compare(Path("bad.cu"), Path("good.cu"), with_ncu=True)
    assert mock_profile.call_count == 2
    assert result["ncu_diff"] is not None
    assert result["ncu_diff"]["schema"] == "ncu_comparison_v1"


def test_bench_compare_ncu_diff_none_when_profile_fails():
    with patch("fournex.bench.compile_kernel", return_value=(True, "")), \
         patch("fournex.bench.time_binary", side_effect=[_timing(10.0), _timing(8.0)]), \
         patch("fournex.bench.profile_with_ncu", return_value=None):
        result = bench_compare(Path("bad.cu"), Path("good.cu"), with_ncu=True)
    assert result["ncu_diff"] is None


# ── module exports ────────────────────────────────────────────────────────────

def test_bench_compare_exported_from_fournex():
    import fournex as fn
    assert hasattr(fn, "bench_compare")
    assert callable(fn.bench_compare)


def test_compile_kernel_exported_from_fournex():
    import fournex as fn
    assert hasattr(fn, "compile_kernel")
    assert callable(fn.compile_kernel)


def test_time_binary_exported_from_fournex():
    import fournex as fn
    assert hasattr(fn, "time_binary")
    assert callable(fn.time_binary)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _run_bench_cmd(argv: list[str], result: dict) -> tuple[int, str]:
    """Run bench_cmd with a mocked bench_compare. Returns (exit_code, stdout)."""
    import io
    from fournex.cli import main

    with patch("fournex.bench.compile_kernel", return_value=(True, "")), \
         patch("fournex.bench.time_binary", side_effect=[_timing(10.0), _timing(8.0)]), \
         patch("fournex.bench.bench_compare", return_value=result) as _mock, \
         patch("sys.stdout", new_callable=io.StringIO) as mock_out:
        exit_code = main(["bench"] + argv)
        output = mock_out.getvalue()
    return exit_code, output


def test_cli_bench_json_schema():
    fake_result = {
        "schema": "frx_bench_v0",
        "arch": "sm_120",
        "before": {"src": "bad.cu", "timing": _timing(10.0)},
        "after":  {"src": "good.cu", "timing": _timing(8.0)},
        "speedup_x": 1.25,
        "ncu_diff": None,
        "compile_errors": [],
    }
    import io
    from fournex.cli import main

    # Write dummy .cu files so path checks pass
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        before = os.path.join(tmp, "bad.cu")
        after  = os.path.join(tmp, "good.cu")
        open(before, "w").close()
        open(after, "w").close()

        with patch("fournex.bench.bench_compare", return_value=fake_result), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            exit_code = main(["bench", before, after, "--json"])
            output = mock_out.getvalue()

    assert exit_code == 0
    parsed = json.loads(output)
    assert parsed["schema"] == "frx_bench_v0"
    assert parsed["speedup_x"] == pytest.approx(1.25)


def test_cli_bench_missing_file_returns_error():
    import io
    from fournex.cli import main

    with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
        exit_code = main(["bench", "nonexistent_before.cu", "nonexistent_after.cu"])
    assert exit_code != 0


def test_cli_bench_with_ncu_and_no_arch_warns():
    import io, tempfile, os
    from fournex.cli import main

    fake_result = {
        "schema": "frx_bench_v0",
        "arch": None,
        "before": {"src": "bad.cu", "timing": _timing(10.0)},
        "after":  {"src": "good.cu", "timing": _timing(8.0)},
        "speedup_x": 1.25,
        "ncu_diff": None,
        "compile_errors": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        before = os.path.join(tmp, "bad.cu")
        after  = os.path.join(tmp, "good.cu")
        open(before, "w").close()
        open(after, "w").close()

        with patch("fournex.bench.bench_compare", return_value=fake_result), \
             patch("sys.stderr", new_callable=io.StringIO) as mock_err, \
             patch("sys.stdout", new_callable=io.StringIO):
            main(["bench", before, after, "--with-ncu"])
            warn = mock_err.getvalue()

    assert "arch" in warn.lower() or "jit" in warn.lower()
