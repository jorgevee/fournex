from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex.cli import StepResult, _compile_ptx_cu, _run_ncu_compare, _load_or_generate_summary


# ---------------------------------------------------------------------------
# _compile_ptx_cu unit tests
# ---------------------------------------------------------------------------

def _fake_cu(tmp_dir: Path) -> Path:
    p = tmp_dir / "test.cu"
    p.write_text("__global__ void k(){}", encoding="utf-8")
    return p


def test_compile_ptx_cu_no_nvcc_returns_step_result():
    with tempfile.TemporaryDirectory() as d:
        cu = _fake_cu(Path(d))
        with patch("fournex.cli.shutil.which", return_value=None):
            result = _compile_ptx_cu(cu)
    assert not result.ok
    assert "nvcc not found" in result.error
    assert result.step == "nvcc -ptx"


def test_compile_ptx_cu_nvcc_failure_captures_reason():
    def fake_which(name):
        return "/usr/bin/nvcc" if name == "nvcc" else None

    def fake_run(cmd, **kwargs):
        class FakeProc:
            returncode = 1
            stderr = "identifier 'foo' is undefined"
        return FakeProc()

    with tempfile.TemporaryDirectory() as d:
        cu = _fake_cu(Path(d))
        with patch("fournex.cli.shutil.which", side_effect=fake_which), \
             patch("fournex.cli.subprocess.run", side_effect=fake_run):
            result = _compile_ptx_cu(cu)
    assert not result.ok
    assert "exited 1" in result.error
    assert "undefined" in result.error


def test_compile_ptx_cu_timeout_returns_step_result():
    def fake_which(name):
        return "/usr/bin/nvcc" if name == "nvcc" else None

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 120)

    with tempfile.TemporaryDirectory() as d:
        cu = _fake_cu(Path(d))
        with patch("fournex.cli.shutil.which", side_effect=fake_which), \
             patch("fournex.cli.subprocess.run", side_effect=fake_run):
            result = _compile_ptx_cu(cu)
    assert not result.ok
    assert "timed out" in result.error


# ---------------------------------------------------------------------------
# _run_ncu_compare unit tests
# ---------------------------------------------------------------------------

def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {"build_flags": "", "with_ncu": True, "gpu_model": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_run_ncu_compare_no_nvcc_returns_step_results():
    with tempfile.TemporaryDirectory() as d:
        cu_a = _fake_cu(Path(d))
        cu_b = Path(d) / "b.cu"
        cu_b.write_text("__global__ void k2(){}", encoding="utf-8")
        with patch("fournex.cli.shutil.which", return_value=None):
            res_a, res_b = _run_ncu_compare(cu_a, cu_b, None, None, _make_args())
    assert not res_a.ok
    assert "nvcc not found" in res_a.error
    assert not res_b.ok


def test_run_ncu_compare_no_ncu_bin_returns_step_results():
    def fake_which(name):
        return "/usr/bin/nvcc" if name == "nvcc" else None

    with tempfile.TemporaryDirectory() as d:
        cu_a = _fake_cu(Path(d))
        cu_b = Path(d) / "b.cu"
        cu_b.write_text("__global__ void k2(){}", encoding="utf-8")
        with patch("fournex.cli.shutil.which", side_effect=fake_which):
            res_a, res_b = _run_ncu_compare(cu_a, cu_b, None, None, _make_args())
    assert not res_a.ok
    assert "ncu not found" in res_a.error
    assert not res_b.ok


# ---------------------------------------------------------------------------
# compare() function tests — calling function directly to test report output
# ---------------------------------------------------------------------------

def _call_compare_with_no_nvcc(cu_a: Path, cu_b: Path) -> tuple[int, str, str]:
    """Run compare() with mocked shutil.which returning None; capture stdout/stderr."""
    import argparse
    from fournex.cli import compare

    args = argparse.Namespace(
        file_a=str(cu_a),
        file_b=str(cu_b),
        label_a=None,
        label_b=None,
        with_ptx=True,
        with_ncu=False,
        ncu_a=None,
        ncu_b=None,
        gpu_model=None,
        arch=None,
        arch_profile=None,
        output_json=False,
        build_flags="",
        before=None,
        after=None,
        before_summary=None,
        after_summary=None,
    )
    # _has_comparison_args must be False (no --before/--after)
    with patch("fournex.cli.shutil.which", return_value=None):
        stdout_cap = io.StringIO()
        stderr_cap = io.StringIO()
        with patch("sys.stdout", stdout_cap), patch("sys.stderr", stderr_cap):
            ret = compare(args)
    return ret, stdout_cap.getvalue(), stderr_cap.getvalue()


def test_compare_with_ptx_no_nvcc_exits_zero():
    with tempfile.TemporaryDirectory() as d:
        cu_a = _fake_cu(Path(d))
        cu_b = Path(d) / "b.cu"
        cu_b.write_text("__global__ void k2(){}", encoding="utf-8")
        ret, stdout, _ = _call_compare_with_no_nvcc(cu_a, cu_b)
    assert ret == 0


def test_compare_with_ptx_no_nvcc_prints_evidence_layer_unavailable():
    with tempfile.TemporaryDirectory() as d:
        cu_a = _fake_cu(Path(d))
        cu_b = Path(d) / "b.cu"
        cu_b.write_text("__global__ void k2(){}", encoding="utf-8")
        ret, stdout, _ = _call_compare_with_no_nvcc(cu_a, cu_b)
    assert "evidence layer unavailable" in stdout
    assert "nvcc not found" in stdout


def test_compare_with_ptx_no_nvcc_json_has_evidence_failures():
    import argparse
    from fournex.cli import compare

    with tempfile.TemporaryDirectory() as d:
        cu_a = _fake_cu(Path(d))
        cu_b = Path(d) / "b.cu"
        cu_b.write_text("__global__ void k2(){}", encoding="utf-8")

        args = argparse.Namespace(
            file_a=str(cu_a),
            file_b=str(cu_b),
            label_a=None, label_b=None,
            with_ptx=True, with_ncu=False,
            ncu_a=None, ncu_b=None,
            gpu_model=None, arch=None, arch_profile=None,
            output_json=True,
            build_flags="", before=None, after=None,
            before_summary=None, after_summary=None,
        )
        stdout_cap = io.StringIO()
        with patch("fournex.cli.shutil.which", return_value=None), \
             patch("sys.stdout", stdout_cap):
            ret = compare(args)

    raw = stdout_cap.getvalue()
    json_start = raw.index("{")  # skip progress lines before JSON
    outer = json.loads(raw[json_start:])
    payload = outer.get("result", outer)

    assert "evidence_failures" in payload
    failures = payload["evidence_failures"]
    ptx_failures = [f for f in failures if f.get("layer") == "ptx"]
    assert len(ptx_failures) == 2
    assert all(f.get("reason") for f in ptx_failures)


# ---------------------------------------------------------------------------
# _load_or_generate_summary: corrupt derived/summary.json warns to stderr
# ---------------------------------------------------------------------------

def test_load_or_generate_summary_corrupt_json_warns():
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        derived = run_dir / "derived"
        derived.mkdir()
        (derived / "summary.json").write_text("{not valid json", encoding="utf-8")
        stderr_capture = io.StringIO()
        with patch("sys.stderr", stderr_capture):
            result = _load_or_generate_summary(run_dir)
    warning_text = stderr_capture.getvalue()
    assert "[warn]" in warning_text
    assert "summary.json" in warning_text
