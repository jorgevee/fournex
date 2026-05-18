"""CLI integration tests for `frx profile`."""
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex.cli import main


# ── Fixtures ──────────────────────────────────────────────────────────────────

_MINIMAL_NCU_CSV = "\n".join([
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "gemm,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,82.0",
    "gemm,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,%,7.0",
    "gemm,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,35.0",
    "gemm,l1tex__t_sector_hit_rate.pct,%,28.0",
    "gemm,sm__issue_active.avg.pct_of_peak_sustained_active,%,55.0",
])

_MINIMAL_PTX = """
.version 8.0
.target sm_80
.address_size 64

.visible .entry simple(.param .u64 p) {
    .reg .f32 %f<4>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd0, [p];
    ld.global.f32 %f0, [%rd0];
    fma.rn.f32 %f1, %f0, %f0, %f0;
    st.global.f32 [%rd0], %f1;
    ret;
}
"""


def _write_temp(content: str, suffix: str) -> Path:
    """Write content to a NamedTemporaryFile and return its path (delete=False)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as f:
        f.write(content)
        return Path(f.name)


# ── --ncu FILE mode ────────────────────────────────────────────────────────────

def test_profile_ncu_csv_exits_zero(capsys) -> None:
    p = _write_temp(_MINIMAL_NCU_CSV, ".csv")
    try:
        rc = main(["profile", "--ncu", str(p)])
        assert rc == 0
    finally:
        p.unlink(missing_ok=True)


def test_profile_ncu_csv_contains_verdict(capsys) -> None:
    p = _write_temp(_MINIMAL_NCU_CSV, ".csv")
    try:
        main(["profile", "--ncu", str(p)])
        out = capsys.readouterr().out
        assert "VERDICT" in out
        assert "bottleneck" in out
    finally:
        p.unlink(missing_ok=True)


def test_profile_ncu_csv_contains_metrics_table(capsys) -> None:
    p = _write_temp(_MINIMAL_NCU_CSV, ".csv")
    try:
        main(["profile", "--ncu", str(p)])
        out = capsys.readouterr().out
        assert "MEASURED METRICS" in out
        assert "DRAM Throughput" in out
        assert "L1 Hit Rate" in out
    finally:
        p.unlink(missing_ok=True)


def test_profile_ncu_csv_contains_recommendations_with_actions(capsys) -> None:
    p = _write_temp(_MINIMAL_NCU_CSV, ".csv")
    try:
        main(["profile", "--ncu", str(p)])
        out = capsys.readouterr().out
        assert "RECOMMENDATIONS" in out
        assert "Actions:" in out
    finally:
        p.unlink(missing_ok=True)


def test_profile_ncu_csv_validate_section_has_ncu_command(capsys) -> None:
    p = _write_temp(_MINIMAL_NCU_CSV, ".csv")
    try:
        main(["profile", "--ncu", str(p)])
        out = capsys.readouterr().out
        assert "Validate:" in out
        assert "ncu --metrics" in out
        assert "--csv ./report.csv ./your_app" in out
    finally:
        p.unlink(missing_ok=True)


def test_profile_ncu_csv_validate_section_has_expected_outcome(capsys) -> None:
    p = _write_temp(_MINIMAL_NCU_CSV, ".csv")
    try:
        main(["profile", "--ncu", str(p)])
        out = capsys.readouterr().out
        # direction arrows and expected-outcome prose should appear
        assert ("<--" in out or "-->" in out)
    finally:
        p.unlink(missing_ok=True)


def test_profile_ncu_json_recs_include_validation_steps(capsys) -> None:
    p = _write_temp(_MINIMAL_NCU_CSV, ".csv")
    try:
        import json
        main(["profile", "--ncu", str(p), "--json"])
        data = json.loads(capsys.readouterr().out)
        recs = data["result"]["recommendations"]
        assert any(len(r.get("validation_steps", [])) > 0 for r in recs), (
            "At least one recommendation should carry validation_steps"
        )
    finally:
        p.unlink(missing_ok=True)


def test_profile_ncu_csv_contains_next_steps(capsys) -> None:
    p = _write_temp(_MINIMAL_NCU_CSV, ".csv")
    try:
        main(["profile", "--ncu", str(p)])
        out = capsys.readouterr().out
        assert "NEXT STEPS" in out
    finally:
        p.unlink(missing_ok=True)


def test_profile_ncu_json_output_is_valid(capsys) -> None:
    p = _write_temp(_MINIMAL_NCU_CSV, ".csv")
    try:
        rc = main(["profile", "--ncu", str(p), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["mode"] == "profile"
        assert "recommendations" in data["result"]
        assert isinstance(data["result"]["recommendations"], list)
    finally:
        p.unlink(missing_ok=True)


# ── --ptx FILE mode ────────────────────────────────────────────────────────────

def test_profile_ptx_exits_zero() -> None:
    p = _write_temp(_MINIMAL_PTX, ".ptx")
    try:
        rc = main(["profile", "--ptx", str(p)])
        assert rc == 0
    finally:
        p.unlink(missing_ok=True)


# ── Error paths ────────────────────────────────────────────────────────────────

def test_profile_missing_ncu_file_exits_one(capsys) -> None:
    rc = main(["profile", "--ncu", "nonexistent_report_xyzzy.csv"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err.lower()


def test_profile_missing_ptx_file_exits_one(capsys) -> None:
    rc = main(["profile", "--ptx", "nonexistent_kernel_xyzzy.ptx"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err.lower()


def test_profile_no_args_exits_one(capsys) -> None:
    rc = main(["profile"])
    assert rc == 1


def test_profile_ncu_not_on_path_exits_one(capsys, monkeypatch) -> None:
    monkeypatch.setattr("fournex.cli.shutil.which", lambda _name: None)
    rc = main(["profile", "--", "my_fake_app"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "ncu" in err.lower()
    assert "frx ncu-command" in err
