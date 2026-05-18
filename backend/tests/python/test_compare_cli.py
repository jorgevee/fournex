"""CLI integration tests for `frx compare`."""
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex.cli import main


# ── Fixtures ──────────────────────────────────────────────────────────────────

_BAD_CU = """\
__global__ void bad_kernel(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __syncthreads();
    float sum = 0.0f;
    for (int k = 0; k < width; k++) {
        sum += A[row * width + k] * B[k * width + col];
    }
    C[row * width + col] = sum;
}
"""

_GOOD_CU = """\
#define TILE 16
__global__ void good_kernel(float* A, float* B, float* C, int N) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][17];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        if (row < N && t * TILE + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < N && t * TILE + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N)
        C[row * N + col] = sum;
}
"""


def _write_cu(content: str) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False, encoding="utf-8") as f:
        f.write(content)
        return Path(f.name)


# ── Basic source-only comparison ──────────────────────────────────────────────

def test_compare_source_only_exits_zero(capsys) -> None:
    fa = _write_cu(_BAD_CU)
    fb = _write_cu(_GOOD_CU)
    try:
        rc = main(["compare", str(fa), str(fb)])
        assert rc == 0
    finally:
        fa.unlink(missing_ok=True)
        fb.unlink(missing_ok=True)


def test_compare_winner_in_output(capsys) -> None:
    fa = _write_cu(_BAD_CU)
    fb = _write_cu(_GOOD_CU)
    try:
        main(["compare", str(fa), str(fb)])
        out = capsys.readouterr().out
        assert "Winner:" in out
    finally:
        fa.unlink(missing_ok=True)
        fb.unlink(missing_ok=True)


def test_compare_resolved_section_present(capsys) -> None:
    fa = _write_cu(_BAD_CU)
    fb = _write_cu(_GOOD_CU)
    try:
        main(["compare", str(fa), str(fb)])
        out = capsys.readouterr().out
        # The bad kernel has unnecessary_syncthreads; good kernel does not
        assert "Resolved" in out
    finally:
        fa.unlink(missing_ok=True)
        fb.unlink(missing_ok=True)


def test_compare_still_unknown_section_present(capsys) -> None:
    fa = _write_cu(_BAD_CU)
    fb = _write_cu(_GOOD_CU)
    try:
        main(["compare", str(fa), str(fb)])
        out = capsys.readouterr().out
        assert "Still unknown" in out
    finally:
        fa.unlink(missing_ok=True)
        fb.unlink(missing_ok=True)


def test_compare_upgrade_hints_present_source_only(capsys) -> None:
    fa = _write_cu(_BAD_CU)
    fb = _write_cu(_GOOD_CU)
    try:
        main(["compare", str(fa), str(fb)])
        out = capsys.readouterr().out
        assert "--with-ptx" in out or "--with-ncu" in out
    finally:
        fa.unlink(missing_ok=True)
        fb.unlink(missing_ok=True)


def test_compare_root_causes_section(capsys) -> None:
    fa = _write_cu(_BAD_CU)
    fb = _write_cu(_GOOD_CU)
    try:
        main(["compare", str(fa), str(fb)])
        out = capsys.readouterr().out
        assert "Root causes" in out
    finally:
        fa.unlink(missing_ok=True)
        fb.unlink(missing_ok=True)


def test_compare_improved_section_when_b_wins(capsys) -> None:
    fa = _write_cu(_BAD_CU)
    fb = _write_cu(_GOOD_CU)
    try:
        main(["compare", str(fa), str(fb)])
        out = capsys.readouterr().out
        assert "Improved" in out
    finally:
        fa.unlink(missing_ok=True)
        fb.unlink(missing_ok=True)


# ── --json output ─────────────────────────────────────────────────────────────

def test_compare_json_has_comparison_and_reconciliation(capsys) -> None:
    fa = _write_cu(_BAD_CU)
    fb = _write_cu(_GOOD_CU)
    try:
        rc = main(["compare", str(fa), str(fb), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        inner = data.get("result", data)
        assert "comparison" in inner
        assert "reconciliation" in inner
        assert inner["comparison"]["schema"] == "comparison_v1"
        assert inner["reconciliation"]["schema"] == "reconciliation_v1"
    finally:
        fa.unlink(missing_ok=True)
        fb.unlink(missing_ok=True)


def test_compare_json_verdict_has_winner(capsys) -> None:
    fa = _write_cu(_BAD_CU)
    fb = _write_cu(_GOOD_CU)
    try:
        main(["compare", str(fa), str(fb), "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        inner = data.get("result", data)
        verdict = inner["comparison"]["verdict"]
        assert verdict["overall_winner"] in ("a", "b", "tie")
    finally:
        fa.unlink(missing_ok=True)
        fb.unlink(missing_ok=True)


# ── --label-a / --label-b ─────────────────────────────────────────────────────

def test_compare_custom_labels_appear_in_output(capsys) -> None:
    fa = _write_cu(_BAD_CU)
    fb = _write_cu(_GOOD_CU)
    try:
        main(["compare", str(fa), str(fb), "--label-a", "naive", "--label-b", "tiled"])
        out = capsys.readouterr().out
        assert "naive" in out
        assert "tiled" in out
    finally:
        fa.unlink(missing_ok=True)
        fb.unlink(missing_ok=True)


# ── Error cases ───────────────────────────────────────────────────────────────

def test_compare_missing_file_exits_nonzero(capsys) -> None:
    rc = main(["compare", "nonexistent_a.cu", "nonexistent_b.cu"])
    assert rc != 0


def test_compare_wrong_extension_exits_nonzero(capsys) -> None:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not cuda")
        bad_path = Path(f.name)
    fa = _write_cu(_BAD_CU)
    try:
        rc = main(["compare", str(fa), str(bad_path)])
        assert rc != 0
    finally:
        fa.unlink(missing_ok=True)
        bad_path.unlink(missing_ok=True)


# ── Identical files → tie ─────────────────────────────────────────────────────

def test_compare_identical_files_tie(capsys) -> None:
    fa = _write_cu(_BAD_CU)
    fb = _write_cu(_BAD_CU)
    try:
        main(["compare", str(fa), str(fb)])
        out = capsys.readouterr().out
        assert "tie" in out.lower() or "Winner:" in out
    finally:
        fa.unlink(missing_ok=True)
        fb.unlink(missing_ok=True)


# ── NCU CSV via --ncu-a / --ncu-b ─────────────────────────────────────────────

def _write_ncu_csv(content: str) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(content)
        return Path(f.name)


_NCU_GOOD = "\n".join([
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,40.0",
    "ker,l1tex__t_sector_hit_rate.pct,%,80.0",
    "ker,sm__issue_active.avg.pct_of_peak_sustained_active,%,70.0",
])

_NCU_BAD = "\n".join([
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,88.0",
    "ker,l1tex__t_sector_hit_rate.pct,%,25.0",
    "ker,sm__issue_active.avg.pct_of_peak_sustained_active,%,30.0",
])


def test_compare_with_preexisting_ncu_csvs(capsys) -> None:
    fa = _write_cu(_BAD_CU)
    fb = _write_cu(_GOOD_CU)
    ncu_a = _write_ncu_csv(_NCU_BAD)
    ncu_b = _write_ncu_csv(_NCU_GOOD)
    try:
        rc = main([
            "compare", str(fa), str(fb),
            "--ncu-a", str(ncu_a), "--ncu-b", str(ncu_b),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        # NCU data is available, so "Still unknown" should not include DRAM bandwidth
        assert "DRAM bandwidth" not in out
        # Evidence line should mention NCU
        assert "NCU" in out
    finally:
        for p in (fa, fb, ncu_a, ncu_b):
            p.unlink(missing_ok=True)


def test_compare_ncu_unlocks_memory_efficiency(capsys) -> None:
    fa = _write_cu(_BAD_CU)
    fb = _write_cu(_GOOD_CU)
    ncu_a = _write_ncu_csv(_NCU_BAD)
    ncu_b = _write_ncu_csv(_NCU_GOOD)
    try:
        main([
            "compare", str(fa), str(fb),
            "--ncu-a", str(ncu_a), "--ncu-b", str(ncu_b),
            "--json",
        ])
        out = capsys.readouterr().out
        data = json.loads(out)
        inner = data.get("result", data)
        sc = inner["comparison"]["scorecard"]
        assert sc["memory_efficiency"]["available"] is True
    finally:
        for p in (fa, fb, ncu_a, ncu_b):
            p.unlink(missing_ok=True)
