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


# ── compare: missing evidence section ────────────────────────────────────────

_STRIDED_KERNEL = """\
__global__ void strided(float* A, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 1024) A[tid * stride] = 1.0f;
}
"""

_SIMPLE_KERNEL = """\
__global__ void simple(float* A) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 1024) A[tid] = 1.0f;
}
"""


def test_compare_shows_missing_evidence_section(capsys) -> None:
    pa = _write_temp(_STRIDED_KERNEL, ".cu")
    pb = _write_temp(_SIMPLE_KERNEL, ".cu")
    try:
        rc = main(["compare", str(pa), str(pb)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Missing evidence" in out
    finally:
        pa.unlink(missing_ok=True)
        pb.unlink(missing_ok=True)


def test_compare_missing_evidence_shows_ncu_command(capsys) -> None:
    pa = _write_temp(_STRIDED_KERNEL, ".cu")
    pb = _write_temp(_SIMPLE_KERNEL, ".cu")
    try:
        main(["compare", str(pa), str(pb)])
        out = capsys.readouterr().out
        assert "ncu --metrics" in out
        assert "ncu --set full" in out
    finally:
        pa.unlink(missing_ok=True)
        pb.unlink(missing_ok=True)


# ── Validation step current_value in CLI output ───────────────────────────────

def test_profile_ncu_validate_shows_was_current_value(capsys) -> None:
    # NCU CSV with tensor core utilization — should show "was 7.0" in Validate block
    p = _write_temp(_MINIMAL_NCU_CSV, ".csv")
    try:
        main(["profile", "--ncu", str(p)])
        out = capsys.readouterr().out
        assert "was 7.0" in out  # tensor_core_utilization_pct from CSV
    finally:
        p.unlink(missing_ok=True)


def test_profile_ncu_json_validation_steps_have_current_value_field(capsys) -> None:
    p = _write_temp(_MINIMAL_NCU_CSV, ".csv")
    try:
        main(["profile", "--ncu", str(p), "--json"])
        data = json.loads(capsys.readouterr().out)
        recs = data["result"]["recommendations"]
        steps_with_current = [
            step for rec in recs
            for step in rec.get("validation_steps", [])
            if step.get("current_value") is not None
        ]
        assert len(steps_with_current) > 0, "At least one validation step should have a current_value"
        # All validation_steps must have the current_value key (may be None)
        for rec in recs:
            for step in rec.get("validation_steps", []):
                assert "current_value" in step
    finally:
        p.unlink(missing_ok=True)


# ── compare --before/--after (evidence mode, validation delta table) ──────────

_BEFORE_NCU_CSV = "\n".join([
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,88.0",
    "ker,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,%,8.0",
    "ker,l1tex__t_sector_hit_rate.pct,%,28.0",
    "ker,lts__t_sector_hit_rate.pct,%,38.0",
    "ker,sm__issue_active.avg.pct_of_peak_sustained_active,%,55.0",
    "ker,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,45.0",
    "ker,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_long_scoreboard,%,22.0",
])

_AFTER_NCU_CSV = "\n".join([
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,42.0",
    "ker,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,%,62.0",
    "ker,l1tex__t_sector_hit_rate.pct,%,72.0",
    "ker,lts__t_sector_hit_rate.pct,%,80.0",
    "ker,sm__issue_active.avg.pct_of_peak_sustained_active,%,75.0",
    "ker,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,8.0",
    "ker,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_long_scoreboard,%,4.0",
])


def test_compare_before_after_ncu_exits_zero(capsys) -> None:
    pb = _write_temp(_BEFORE_NCU_CSV, ".csv")
    pa = _write_temp(_AFTER_NCU_CSV, ".csv")
    try:
        rc = main(["compare", "--before", str(pb), "--after", str(pa)])
        assert rc == 0
    finally:
        pb.unlink(missing_ok=True)
        pa.unlink(missing_ok=True)


def test_compare_before_after_ncu_shows_validation_header(capsys) -> None:
    pb = _write_temp(_BEFORE_NCU_CSV, ".csv")
    pa = _write_temp(_AFTER_NCU_CSV, ".csv")
    try:
        main(["compare", "--before", str(pb), "--after", str(pa)])
        out = capsys.readouterr().out
        assert "BEFORE / AFTER VALIDATION" in out
    finally:
        pb.unlink(missing_ok=True)
        pa.unlink(missing_ok=True)


def test_compare_before_after_ncu_shows_result_line(capsys) -> None:
    pb = _write_temp(_BEFORE_NCU_CSV, ".csv")
    pa = _write_temp(_AFTER_NCU_CSV, ".csv")
    try:
        main(["compare", "--before", str(pb), "--after", str(pa)])
        out = capsys.readouterr().out
        assert "Result:" in out
    finally:
        pb.unlink(missing_ok=True)
        pa.unlink(missing_ok=True)


def test_compare_before_after_ncu_shows_metric_labels(capsys) -> None:
    pb = _write_temp(_BEFORE_NCU_CSV, ".csv")
    pa = _write_temp(_AFTER_NCU_CSV, ".csv")
    try:
        main(["compare", "--before", str(pb), "--after", str(pa)])
        out = capsys.readouterr().out
        assert "DRAM Throughput" in out
        assert "L1 Hit Rate" in out
    finally:
        pb.unlink(missing_ok=True)
        pa.unlink(missing_ok=True)


def test_compare_before_after_no_positionals_error(capsys) -> None:
    """compare with no positionals and no --before/--after flags exits 1."""
    rc = main(["compare"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "frx compare" in err
