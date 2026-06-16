"""Tests for the case-study harness (static path, no GPU required)."""
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
REPO = Path(__file__).resolve().parents[3]
ZOO = REPO / "demos" / "cuda_zoo"

from fournex.case_study import (
    CaseExpectation,
    discover_case_studies,
    emit_case_study_artifacts,
    load_case_study,
    render_case_study_readme,
    render_case_study_txt,
    run_case_study,
    validate_case_study,
)

pytestmark = pytest.mark.skipif(not ZOO.exists(), reason="demos/cuda_zoo not present")


def _case(name: str):
    return load_case_study(ZOO / name)


# ── Discovery / loading ───────────────────────────────────────────────────────

def test_discover_finds_all_zoo_cases():
    names = {c.name for c in discover_case_studies(ZOO)}
    assert {
        "uncoalesced_global_loads",
        "naive_vs_tiled_matmul",
        "excess_synchronization",
        "register_pressure",
    } <= names


def test_load_case_study_fields():
    case = _case("01_uncoalesced")
    assert case.name == "uncoalesced_global_loads"
    assert case.bad_source == "bad.cu"
    assert "uncoalesced_access" in case.expected.detected_before


def test_load_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_case_study(tmp_path)


# ── Run + validation (the real proof loop) ────────────────────────────────────

@pytest.mark.parametrize("case_dir", [
    "01_uncoalesced", "02_matmul_notiled", "03_excess_sync", "04_register_pressure",
])
def test_each_zoo_case_validates(case_dir):
    result = run_case_study(_case(case_dir))
    assert result["validation"]["passed"], result["validation"]
    # every expected-resolved finding must actually be resolved
    case = _case(case_dir)
    assert set(case.expected.resolved_after) <= set(result["validation"]["resolved"])


def test_uncoalesced_detected_and_resolved():
    result = run_case_study(_case("01_uncoalesced"))
    assert "uncoalesced_access" in result["before_codes"]
    assert "uncoalesced_access" in result["validation"]["resolved"]
    assert result["validation"]["new"] == []


def test_validation_fails_when_expected_finding_absent():
    case = replace(
        _case("01_uncoalesced"),
        expected=CaseExpectation(detected_before=["nonexistent_finding"], resolved_after=[]),
    )
    result = run_case_study(case)
    assert result["validation"]["passed"] is False
    detect = next(c for c in result["validation"]["checks"]
                  if c["check"] == "expected_bottlenecks_detected_before")
    assert detect["status"] == "fail"
    assert "nonexistent_finding" in detect["missing"]


def test_validation_fails_when_resolution_not_achieved():
    # missing_vectorized_loads persists in good.cu, so it is never "resolved"
    case = replace(
        _case("01_uncoalesced"),
        expected=CaseExpectation(
            detected_before=["uncoalesced_access"],
            resolved_after=["missing_vectorized_loads"],
        ),
    )
    result = run_case_study(case)
    assert result["validation"]["passed"] is False


# ── Rendering + artifacts ─────────────────────────────────────────────────────

def test_transcript_contains_verdict():
    txt = render_case_study_txt(run_case_study(_case("01_uncoalesced")))
    assert "CASE STUDY:" in txt
    assert "VALIDATED" in txt


def test_readme_renders_status():
    md = render_case_study_readme(run_case_study(_case("02_matmul_notiled")))
    assert md.startswith("# Case Study:")
    assert "validated" in md


def test_emit_artifacts_writes_bundle(tmp_path):
    result = run_case_study(_case("03_excess_sync"))
    written = emit_case_study_artifacts(result, tmp_path, emit_readme=True)
    for name in ("case_study.txt", "diagnosis.txt", "llm_brief.txt",
                 "evidence.json", "compare.json", "validation.json", "README.md"):
        assert name in written
        assert (tmp_path / name).exists()
        assert (tmp_path / name).read_text(encoding="utf-8").strip()
