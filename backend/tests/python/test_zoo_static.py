"""Static analysis smoke tests for the CUDA antipattern zoo.

Each bad.cu should trigger specific rule IDs.
Each good.cu should NOT trigger those same rules.
"""
import sys
from pathlib import Path

import pytest

ROOT   = Path(__file__).resolve().parents[2]
ZOO    = ROOT.parent / "demos" / "cuda_zoo"
sys.path.insert(0, str(ROOT / "python"))

import fournex as fn


# ── helpers ──────────────────────────────────────────────────────────────────

def _findings(cu_path: Path) -> set[str]:
    src = cu_path.read_text(encoding="utf-8")
    result = fn.inspect_cuda_source(src)
    codes = set()
    for k in result.get("kernels", []):
        for f in k.get("findings", []):
            codes.add(f["code"])
    for f in result.get("launch_findings", []):
        codes.add(f["code"])
    return codes


def _zoo(subdir: str, filename: str) -> Path:
    p = ZOO / subdir / filename
    if not p.exists():
        pytest.skip(f"Zoo file not found: {p}")
    return p


# ── 01: uncoalesced access ────────────────────────────────────────────────────

def test_uncoalesced_bad_flags_uncoalesced_access():
    codes = _findings(_zoo("01_uncoalesced", "bad.cu"))
    assert "uncoalesced_access" in codes, f"Expected uncoalesced_access, got: {codes}"


def test_uncoalesced_good_does_not_flag_uncoalesced_access():
    codes = _findings(_zoo("01_uncoalesced", "good.cu"))
    assert "uncoalesced_access" not in codes, (
        f"good.cu should not flag uncoalesced_access, got: {codes}"
    )


# ── 02: naive GEMM ────────────────────────────────────────────────────────────

def test_matmul_bad_flags_fp32_only_matmul():
    codes = _findings(_zoo("02_matmul_notiled", "bad.cu"))
    assert "fp32_only_matmul" in codes, f"Expected fp32_only_matmul, got: {codes}"


def test_matmul_bad_flags_no_shared_memory_tiling():
    codes = _findings(_zoo("02_matmul_notiled", "bad.cu"))
    assert "no_shared_memory_tiling" in codes, (
        f"Expected no_shared_memory_tiling, got: {codes}"
    )


def test_matmul_good_does_not_flag_no_shared_memory_tiling():
    codes = _findings(_zoo("02_matmul_notiled", "good.cu"))
    assert "no_shared_memory_tiling" not in codes, (
        f"good.cu should not flag no_shared_memory_tiling, got: {codes}"
    )


# ── 03: excess sync ───────────────────────────────────────────────────────────

def test_excess_sync_bad_flags_sync_inside_tight_loop():
    codes = _findings(_zoo("03_excess_sync", "bad.cu"))
    assert "sync_inside_tight_loop" in codes, (
        f"Expected sync_inside_tight_loop, got: {codes}"
    )


def test_excess_sync_good_does_not_flag_sync_inside_tight_loop():
    codes = _findings(_zoo("03_excess_sync", "good.cu"))
    assert "sync_inside_tight_loop" not in codes, (
        f"good.cu should not flag sync_inside_tight_loop, got: {codes}"
    )


# ── 04: register pressure ─────────────────────────────────────────────────────

def test_register_pressure_bad_flags_high_register_pressure():
    codes = _findings(_zoo("04_register_pressure", "bad.cu"))
    assert "high_register_pressure" in codes, (
        f"Expected high_register_pressure, got: {codes}"
    )


def test_register_pressure_good_does_not_flag_high_register_pressure():
    codes = _findings(_zoo("04_register_pressure", "good.cu"))
    assert "high_register_pressure" not in codes, (
        f"good.cu should not flag high_register_pressure, got: {codes}"
    )
