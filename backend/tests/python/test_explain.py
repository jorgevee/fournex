"""Tests for frx explain — LLM-ready optimization brief pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as fn
from fournex.explain import (
    build_explain_result,
    render_evidence_json,
    render_llm_prompt_txt,
    render_summary_txt,
)
from fournex.ncu_analysis import analyze_ncu_csv_text

FIXTURES = ROOT / "tests" / "evals" / "fixtures"

_UNCOALESCED_CSV = (FIXTURES / "uncoalesced_dram_bound.csv").read_text()
_TENSOR_IDLE_CSV = (FIXTURES / "tensor_core_idle.csv").read_text()
_EXCESSIVE_SYNC_CSV = (FIXTURES / "excessive_sync.csv").read_text()
_REGISTER_CSV = (FIXTURES / "register_pressure.csv").read_text()
_WELL_OPTIMIZED_CSV = (FIXTURES / "well_optimized.csv").read_text()

_STRIDED_KERNEL_SRC = """\
__global__ void k(const float* __restrict__ src, float* __restrict__ dst, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    dst[tid] = src[tid * stride];
}
"""


# ── build_explain_result ──────────────────────────────────────────────────────

def test_explain_result_has_required_schema_fields():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    for field in ("schema", "layers_available", "primary_diagnosis",
                  "diagnoses", "key_metrics", "static_findings",
                  "ncu_bottlenecks", "top_recommendations", "missing_data"):
        assert field in result, f"missing field: {field}"
    assert result["schema"] == "frx_explain_v0"


def test_explain_result_ncu_only_layers_available():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    assert "ncu" in result["layers_available"]
    assert "source" not in result["layers_available"]


def test_explain_result_with_static_includes_source_layer():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    static = fn.inspect_cuda_source(_STRIDED_KERNEL_SRC)
    result = build_explain_result(ncu_result=ncu, static_result=static)
    assert "source" in result["layers_available"]
    assert "ncu" in result["layers_available"]


def test_explain_result_with_static_includes_findings():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    static = fn.inspect_cuda_source(_STRIDED_KERNEL_SRC)
    result = build_explain_result(ncu_result=ncu, static_result=static)
    assert isinstance(result["static_findings"], list)


def test_explain_result_primary_diagnosis_set_for_bottleneck():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    assert result["primary_diagnosis"] is not None


def test_explain_result_key_metrics_contain_dram():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    assert "avg_dram_throughput_pct" in result["key_metrics"]
    assert result["key_metrics"]["avg_dram_throughput_pct"] == 89.0


def test_explain_result_ncu_bottlenecks_have_label_and_score():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    for b in result["ncu_bottlenecks"]:
        assert "label" in b
        assert "score" in b
        assert 0.0 <= b["score"] <= 1.0


def test_explain_result_missing_data_includes_ptx():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    layers = [m["layer"] for m in result["missing_data"]]
    assert "ptx" in layers


def test_explain_result_missing_data_includes_source_when_not_provided():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    layers = [m["layer"] for m in result["missing_data"]]
    assert "source" in layers


def test_explain_result_missing_data_no_source_entry_when_provided():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    static = fn.inspect_cuda_source(_STRIDED_KERNEL_SRC)
    result = build_explain_result(ncu_result=ncu, static_result=static)
    layers = [m["layer"] for m in result["missing_data"]]
    assert "source" not in layers


def test_explain_result_well_optimized_has_no_bottlenecks():
    ncu = analyze_ncu_csv_text(_WELL_OPTIMIZED_CSV)
    result = build_explain_result(ncu_result=ncu)
    assert result["ncu_bottlenecks"] == [] or all(
        b["score"] < 0.3 for b in result["ncu_bottlenecks"]
    )


# ── render_summary_txt ────────────────────────────────────────────────────────

def test_summary_includes_primary_issue():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_summary_txt(result, ncu_filename="uncoalesced_dram_bound.csv")
    assert "PRIMARY ISSUE" in txt


def test_summary_includes_evidence_section():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_summary_txt(result)
    assert "EVIDENCE" in txt


def test_summary_includes_ncu_metric_values():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_summary_txt(result)
    assert "89.0" in txt or "89" in txt  # DRAM throughput value


def test_summary_includes_what_to_fix():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_summary_txt(result)
    assert "WHAT TO FIX" in txt


def test_summary_includes_missing_data_section():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_summary_txt(result)
    assert "MISSING DATA" in txt


def test_summary_no_src_omits_source_evidence():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_summary_txt(result)
    assert "[Source]" not in txt


def test_summary_with_src_includes_source_evidence():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    static = fn.inspect_cuda_source(_STRIDED_KERNEL_SRC)
    result = build_explain_result(ncu_result=ncu, static_result=static)
    if result["static_findings"]:
        txt = render_summary_txt(result)
        assert "[Source]" in txt


def test_summary_ncu_filename_appears_in_header():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_summary_txt(result, ncu_filename="my_report.csv")
    assert "my_report.csv" in txt


# ── render_llm_prompt_txt ─────────────────────────────────────────────────────

def test_prompt_includes_guardrail_text():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_llm_prompt_txt(result)
    assert "Do NOT rewrite the entire kernel" in txt


def test_prompt_includes_preserve_correctness_rule():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_llm_prompt_txt(result)
    assert "Preserve correctness" in txt


def test_prompt_includes_primary_bottleneck_header():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_llm_prompt_txt(result)
    assert "PRIMARY BOTTLENECK" in txt


def test_prompt_includes_specific_question_section():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_llm_prompt_txt(result)
    assert "SPECIFIC QUESTION" in txt


def test_prompt_includes_metrics_table():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_llm_prompt_txt(result)
    assert "ALL PROFILER METRICS" in txt or "DRAM" in txt


def test_prompt_includes_kernel_source_when_provided():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_llm_prompt_txt(result, kernel_source=_STRIDED_KERNEL_SRC)
    assert "tid * stride" in txt or "__global__" in txt


def test_prompt_no_source_shows_not_provided():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_llm_prompt_txt(result, kernel_source=None)
    assert "Not provided" in txt or "not provided" in txt


def test_prompt_excessive_sync_has_sync_specific_question():
    ncu = analyze_ncu_csv_text(_EXCESSIVE_SYNC_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_llm_prompt_txt(result)
    # Should mention barriers or syncthreads
    assert "__syncthreads" in txt or "barrier" in txt.lower() or "sync" in txt.lower()


def test_prompt_tensor_core_idle_has_tensor_specific_question():
    ncu = analyze_ncu_csv_text(_TENSOR_IDLE_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_llm_prompt_txt(result)
    assert "tensor" in txt.lower() or "TENSOR" in txt


def test_prompt_includes_re_profile_note():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_llm_prompt_txt(result)
    assert "re-profile" in txt.lower() or "re-run" in txt.lower() or "Nsight" in txt


# ── render_evidence_json ──────────────────────────────────────────────────────

def test_evidence_json_is_valid_json():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    raw = render_evidence_json(result)
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)


def test_evidence_json_has_schema_field():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    parsed = json.loads(render_evidence_json(result))
    assert parsed["schema"] == "frx_explain_v0"


def test_evidence_json_has_key_metrics():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    parsed = json.loads(render_evidence_json(result))
    assert "key_metrics" in parsed
    assert isinstance(parsed["key_metrics"], dict)


def test_evidence_json_roundtrips_ncu_bottlenecks():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    parsed = json.loads(render_evidence_json(result))
    assert isinstance(parsed["ncu_bottlenecks"], list)
    if parsed["ncu_bottlenecks"]:
        assert "label" in parsed["ncu_bottlenecks"][0]


# ── enriched top_recommendations ─────────────────────────────────────────────

def test_top_recommendations_include_speedup_fields():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    for rec in result["top_recommendations"]:
        assert "estimated_speedup_pct_min" in rec
        assert "estimated_speedup_pct_max" in rec


def test_top_recommendations_include_validation_steps():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    for rec in result["top_recommendations"]:
        assert "validation_steps" in rec
        assert isinstance(rec["validation_steps"], list)


def test_top_recommendations_include_why():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    for rec in result["top_recommendations"]:
        assert "why" in rec
        assert isinstance(rec["why"], str)


def test_prompt_includes_expected_improvement_section():
    ncu = analyze_ncu_csv_text(_UNCOALESCED_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_llm_prompt_txt(result)
    assert "EXPECTED IMPROVEMENT" in txt


def test_prompt_speedup_range_shown_when_catalog_has_estimates():
    ncu = analyze_ncu_csv_text(_EXCESSIVE_SYNC_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_llm_prompt_txt(result)
    # excessive_sync fires rec_ncu_reduce_syncthreads which has estimated_speedup_pct_min/max
    if any(
        r.get("estimated_speedup_pct_min") is not None
        for r in result["top_recommendations"]
    ):
        assert "%" in txt.split("EXPECTED IMPROVEMENT")[1].split("EVIDENCE")[0]


def test_summary_what_to_fix_shows_speedup_when_available():
    ncu = analyze_ncu_csv_text(_EXCESSIVE_SYNC_CSV)
    result = build_explain_result(ncu_result=ncu)
    txt = render_summary_txt(result)
    recs_with_speedup = [
        r for r in result["top_recommendations"]
        if r.get("estimated_speedup_pct_min") is not None
    ]
    if recs_with_speedup:
        assert "est." in txt


# ── fn module exports ─────────────────────────────────────────────────────────

def test_build_explain_result_exported_from_fournex():
    assert hasattr(fn, "build_explain_result")
    assert callable(fn.build_explain_result)


def test_render_functions_exported_from_fournex():
    assert hasattr(fn, "render_summary_txt")
    assert hasattr(fn, "render_llm_prompt_txt")
    assert hasattr(fn, "render_evidence_json")
