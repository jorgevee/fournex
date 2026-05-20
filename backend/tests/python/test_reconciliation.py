import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex.reconciliation import reconcile_evidence, what_evidence_is_missing


# ── Fixture builders ──────────────────────────────────────────────────────────

def _static(finding_codes: list[str], access_styles: list[str] | None = None) -> dict:
    findings = [{"code": c, "severity": "medium", "line": 1} for c in finding_codes]
    styles = access_styles or []
    return {
        "schema_version": "cuda_static_v1",
        "findings": findings,
        "kernels": [{"name": "k", "memory_access_styles": styles, "findings": findings}],
        "launches": [],
        "launch_advisor": {},
    }


def _ncu_csv(*rows: str) -> str:
    return "\n".join(["Kernel Name,Metric Name,Metric Unit,Metric Value"] + list(rows))


def _ncu_result(bottleneck_labels: list[str], metrics: dict | None = None) -> dict:
    """Build a minimal ncu result dict that reconcile_evidence can consume."""
    m = metrics or {}
    ncu_run_summary = {
        "kernel_count": 1,
        "kernels_with_ncu_data": 1,
        "avg_dram_throughput_pct": m.get("dram", 50.0),
        "avg_tensor_core_utilization_pct": m.get("tc", 50.0),
        "avg_l1_cache_hit_rate_pct": m.get("l1", 70.0),
        "avg_l2_cache_hit_rate_pct": m.get("l2", 70.0),
        "avg_global_load_sectors_per_request": m.get("sectors", 1.0),
        "avg_issue_slot_utilization_pct": m.get("isu", 70.0),
        "avg_occupancy_pct": m.get("occ", 70.0),
        "avg_eligible_warps_per_scheduler": None,
        "avg_scheduler_active_pct": None,
        "avg_registers_per_thread": None,
        "avg_shared_memory_per_block_bytes": None,
        "avg_threads_per_block": None,
        "occupancy_limiting_factor_counts": {},
        "occupancy_limit_causes": m.get("occ_causes", []),
        "dominant_warp_stall": m.get("stall", "unknown"),
        "dominant_warp_stall_pct": 0.0,
        "warp_stall_breakdown": {},
        "kernels_with_warp_stall_data": 0,
        "memory_stall_fraction": m.get("mem_stall", 0.0),
        "compute_stall_fraction": 0.0,
    }
    bottlenecks = [{"label": lbl, "score": 0.8, "evidence": {}, "worst_steps": []} for lbl in bottleneck_labels]
    return {
        "schema": "ncu_analysis_v1",
        "ncu_run_summary": ncu_run_summary,
        "bottlenecks": bottlenecks,
    }


_PTX_SPILL = """
.version 8.0
.target sm_80
.visible .entry heavy_kernel() {
    .reg .f32 %f<130>;
    .reg .b64 %SP;
    .local .align 4 .b8 __local_depot0[512];
    ld.local.f32 %f0, [%SP+0];
    st.local.f32 [%SP+8], %f0;
    ld.global.f32 %f1, [%SP+0];
    ld.global.f32 %f2, [%SP+4];
    ld.global.f32 %f3, [%SP+8];
    ld.global.f32 %f4, [%SP+12];
    ret;
}
"""

_PTX_BRANCH = """
.version 8.0
.target sm_80
.visible .entry branch_kernel() {
    .reg .pred %p<4>;
    .reg .f32 %f<32>;
    .reg .b32 %r<8>;
    ld.global.f32 %f0, [%r0];
    setp.lt.f32 %p0, %f0, 0f00000000;
    @%p0 bra $L_branch1;
    fma.rn.f32 %f1, %f0, %f0, %f0;
    @%p1 bra $L_branch2;
    fma.rn.f32 %f2, %f1, %f1, %f1;
    @%p2 bra $L_branch3;
    bra $L_end;
$L_branch1: fma.rn.f32 %f1, %f0, %f0, %f0;
$L_branch2: fma.rn.f32 %f2, %f1, %f1, %f1;
$L_branch3: fma.rn.f32 %f3, %f2, %f2, %f2;
    bra $L_branch1;
$L_end: ret;
}
"""

_PTX_GLOBAL_HEAVY = """
.version 8.0
.target sm_80
.visible .entry global_kernel() {
    .reg .f32 %f<32>;
    .reg .b64 %rd<8>;
    ld.global.f32 %f0, [%rd0];
    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd2];
    ld.global.f32 %f3, [%rd3];
    ld.global.f32 %f4, [%rd4];
    ld.global.f32 %f5, [%rd5];
    ld.global.f32 %f6, [%rd6];
    ld.global.f32 %f7, [%rd7];
    ld.global.f32 %f8, [%rd0];
    ld.global.f32 %f9, [%rd1];
    ld.global.f32 %f10, [%rd2];
    ld.global.f32 %f11, [%rd3];
    st.global.f32 [%rd0], %f0;
    st.global.f32 [%rd1], %f1;
    st.global.f32 [%rd2], %f2;
    st.global.f32 [%rd3], %f3;
    fma.rn.f32 %f12, %f0, %f1, %f2;
    fma.rn.f32 %f13, %f3, %f4, %f5;
    fma.rn.f32 %f14, %f6, %f7, %f8;
    fma.rn.f32 %f15, %f9, %f10, %f11;
    fma.rn.f32 %f16, %f12, %f13, %f14;
    fma.rn.f32 %f17, %f15, %f16, %f0;
    ret;
}
"""


def _ptx_result(ptx_text: str) -> dict:
    from fournex.ptx_analysis import analyze_ptx_text
    return analyze_ptx_text(ptx_text)


def _profiler_result(bottleneck_labels: list[str]) -> dict:
    return {
        "bottlenecks": [{"label": lbl, "score": 0.8, "evidence": {}, "worst_steps": []} for lbl in bottleneck_labels],
    }


# ── Schema and structure ──────────────────────────────────────────────────────

def test_empty_input_returns_valid_schema() -> None:
    result = reconcile_evidence()
    assert result["schema"] == "reconciliation_v1"
    assert result["layers_available"] == []
    assert result["diagnoses"] == []
    assert result["unreconciled"] == {}


def test_layers_available_reflects_inputs() -> None:
    result = reconcile_evidence(static=_static([]), ncu=_ncu_result([]))
    assert "source" in result["layers_available"]
    assert "ncu" in result["layers_available"]
    assert "ptx" not in result["layers_available"]
    assert "profiler" not in result["layers_available"]


def test_diagnosis_has_required_fields() -> None:
    result = reconcile_evidence(
        static=_static(["strided_or_pitched"], ["strided_or_pitched"]),
    )
    assert result["diagnoses"]
    d = result["diagnoses"][0]
    for key in ("label", "display_name", "confidence", "severity",
                "layers_confirming", "evidence", "fix_summary", "recommendation_ids"):
        assert key in d, f"missing key: {key}"


# ── Diagnosis: inefficient_global_memory_access ───────────────────────────────

def test_inefficient_global_source_only() -> None:
    result = reconcile_evidence(static=_static(["strided_or_pitched"], ["strided_or_pitched"]))
    labels = {d["label"] for d in result["diagnoses"]}
    assert "inefficient_global_memory_access" in labels


def test_inefficient_global_source_style_also_fires() -> None:
    # style present, finding code absent
    result = reconcile_evidence(static=_static([], ["strided_or_pitched"]))
    labels = {d["label"] for d in result["diagnoses"]}
    assert "inefficient_global_memory_access" in labels


def test_inefficient_global_ncu_only() -> None:
    ncu = _ncu_result(["uncoalesced_access"], {"sectors": 9.0})
    result = reconcile_evidence(ncu=ncu)
    labels = {d["label"] for d in result["diagnoses"]}
    assert "inefficient_global_memory_access" in labels


def test_inefficient_global_source_plus_ncu_confidence_high() -> None:
    ncu = _ncu_result(["uncoalesced_access"], {"sectors": 9.0})
    result = reconcile_evidence(
        static=_static(["strided_or_pitched"], ["strided_or_pitched"]),
        ncu=ncu,
    )
    d = next(d for d in result["diagnoses"] if d["label"] == "inefficient_global_memory_access")
    assert d["confidence"] == "high"
    assert "source" in d["layers_confirming"]
    assert "ncu" in d["layers_confirming"]


def test_inefficient_global_ncu_evidence_has_sectors_metric() -> None:
    ncu = _ncu_result(["uncoalesced_access"], {"sectors": 10.4})
    result = reconcile_evidence(ncu=ncu)
    d = next(d for d in result["diagnoses"] if d["label"] == "inefficient_global_memory_access")
    assert d["evidence"]["ncu"]["metrics"].get("avg_global_load_sectors_per_request") == pytest_approx(10.4)


# ── Diagnosis: excessive_synchronization ─────────────────────────────────────

def test_excessive_sync_source_unnecessary() -> None:
    result = reconcile_evidence(static=_static(["unnecessary_syncthreads"]))
    labels = {d["label"] for d in result["diagnoses"]}
    assert "excessive_synchronization" in labels


def test_excessive_sync_source_conditional() -> None:
    result = reconcile_evidence(static=_static(["conditional_syncthreads"]))
    labels = {d["label"] for d in result["diagnoses"]}
    assert "excessive_synchronization" in labels


def test_excessive_sync_ncu_warp_stall() -> None:
    ncu = _ncu_result(["warp_stall_sync"], {"stall": "barrier"})
    result = reconcile_evidence(ncu=ncu)
    labels = {d["label"] for d in result["diagnoses"]}
    assert "excessive_synchronization" in labels


def test_excessive_sync_profiler_confirms() -> None:
    result = reconcile_evidence(profiler=_profiler_result(["sync_bound"]))
    labels = {d["label"] for d in result["diagnoses"]}
    assert "excessive_synchronization" in labels


def test_excessive_sync_source_plus_profiler_confidence_high() -> None:
    result = reconcile_evidence(
        static=_static(["unnecessary_syncthreads"]),
        profiler=_profiler_result(["sync_bound"]),
    )
    d = next(d for d in result["diagnoses"] if d["label"] == "excessive_synchronization")
    assert d["confidence"] == "high"


def test_excessive_sync_one_of_three_layers_low_medium() -> None:
    result = reconcile_evidence(
        static=_static(["unnecessary_syncthreads"]),
        ncu=_ncu_result([]),  # ncu present but no warp_stall_sync
        profiler=_profiler_result([]),
    )
    d = next((d for d in result["diagnoses"] if d["label"] == "excessive_synchronization"), None)
    assert d is not None
    assert d["confidence"] == "low-medium"


# ── Diagnosis: register_pressure ──────────────────────────────────────────────

def test_register_pressure_ptx_spills() -> None:
    result = reconcile_evidence(ptx=_ptx_result(_PTX_SPILL))
    labels = {d["label"] for d in result["diagnoses"]}
    assert "register_pressure" in labels


def test_register_pressure_ncu_occupancy_limited() -> None:
    ncu = _ncu_result(["occupancy_limited_by_registers"], {"occ_causes": ["registers"], "occ": 25.0})
    result = reconcile_evidence(ncu=ncu)
    labels = {d["label"] for d in result["diagnoses"]}
    assert "register_pressure" in labels


def test_register_pressure_ptx_plus_ncu_confidence_high() -> None:
    ncu = _ncu_result(["occupancy_limited_by_registers"], {"occ_causes": ["registers"]})
    result = reconcile_evidence(ptx=_ptx_result(_PTX_SPILL), ncu=ncu)
    d = next(d for d in result["diagnoses"] if d["label"] == "register_pressure")
    assert d["confidence"] == "high"
    assert "ptx" in d["layers_confirming"]
    assert "ncu" in d["layers_confirming"]


# ── Diagnosis: tensor_core_underutilization ───────────────────────────────────

def test_tensor_core_ncu_only() -> None:
    ncu = _ncu_result(["tensor_core_underutilized"], {"tc": 5.0})
    result = reconcile_evidence(ncu=ncu)
    labels = {d["label"] for d in result["diagnoses"]}
    assert "tensor_core_underutilization" in labels


def test_tensor_core_absent_when_tc_high() -> None:
    ncu = _ncu_result([], {"tc": 80.0})
    result = reconcile_evidence(ncu=ncu)
    labels = {d["label"] for d in result["diagnoses"]}
    assert "tensor_core_underutilization" not in labels


# ── Diagnosis: memory_bandwidth_saturation ────────────────────────────────────

def test_bandwidth_saturation_ncu_only() -> None:
    ncu = _ncu_result(["memory_bandwidth_bound"], {"dram": 85.0})
    result = reconcile_evidence(ncu=ncu)
    labels = {d["label"] for d in result["diagnoses"]}
    assert "memory_bandwidth_saturation" in labels


def test_bandwidth_saturation_ptx_global_heavy() -> None:
    result = reconcile_evidence(ptx=_ptx_result(_PTX_GLOBAL_HEAVY))
    labels = {d["label"] for d in result["diagnoses"]}
    assert "memory_bandwidth_saturation" in labels


def test_bandwidth_saturation_ptx_plus_ncu_medium_high() -> None:
    ncu = _ncu_result(["memory_bandwidth_bound"], {"dram": 85.0})
    # 3 layers available (ptx + ncu + profiler), 2 confirming -> medium-high
    result = reconcile_evidence(
        ptx=_ptx_result(_PTX_GLOBAL_HEAVY),
        ncu=ncu,
        profiler=_profiler_result([]),
    )
    d = next((d for d in result["diagnoses"] if d["label"] == "memory_bandwidth_saturation"), None)
    assert d is not None
    assert d["confidence"] in ("high", "medium-high")


# ── Diagnosis: warp_divergence_risk ──────────────────────────────────────────

def test_warp_divergence_ptx_branch_heavy() -> None:
    result = reconcile_evidence(ptx=_ptx_result(_PTX_BRANCH))
    labels = {d["label"] for d in result["diagnoses"]}
    assert "warp_divergence_risk" in labels


def test_warp_divergence_absent_when_no_branch() -> None:
    result = reconcile_evidence(ptx=_ptx_result(_PTX_GLOBAL_HEAVY))
    labels = {d["label"] for d in result["diagnoses"]}
    assert "warp_divergence_risk" not in labels


# ── Confidence algorithm ──────────────────────────────────────────────────────

def test_single_layer_single_available_is_medium() -> None:
    result = reconcile_evidence(static=_static(["unnecessary_syncthreads"]))
    d = next(d for d in result["diagnoses"] if d["label"] == "excessive_synchronization")
    assert d["confidence"] == "medium"


def test_single_layer_two_available_is_low_medium() -> None:
    result = reconcile_evidence(
        static=_static(["unnecessary_syncthreads"]),
        profiler=_profiler_result([]),  # profiler available but not confirming
    )
    d = next(d for d in result["diagnoses"] if d["label"] == "excessive_synchronization")
    assert d["confidence"] == "low-medium"


def test_three_layers_confirming_is_confirmed() -> None:
    ncu = _ncu_result(["warp_stall_sync"], {"stall": "barrier"})
    result = reconcile_evidence(
        static=_static(["unnecessary_syncthreads"]),
        ncu=ncu,
        profiler=_profiler_result(["sync_bound"]),
    )
    d = next(d for d in result["diagnoses"] if d["label"] == "excessive_synchronization")
    assert d["confidence"] == "confirmed"
    assert len(d["layers_confirming"]) == 3


# ── Unreconciled section ──────────────────────────────────────────────────────

def test_unreconciled_source_findings_not_in_catalog() -> None:
    result = reconcile_evidence(static=_static(["missing_obvious_bounds_guard"]))
    unc = result["unreconciled"].get("source", [])
    assert "missing_obvious_bounds_guard" in unc


def test_unreconciled_ncu_bottlenecks_not_claimed() -> None:
    ncu = _ncu_result(["l1_cache_thrashing"])
    result = reconcile_evidence(ncu=ncu)
    unc = result["unreconciled"].get("ncu", [])
    assert "l1_cache_thrashing" in unc


def test_claimed_codes_not_in_unreconciled() -> None:
    ncu = _ncu_result(["uncoalesced_access"], {"sectors": 9.0})
    result = reconcile_evidence(ncu=ncu)
    unc = result["unreconciled"].get("ncu", [])
    assert "uncoalesced_access" not in unc


def test_empty_unreconciled_when_all_claimed() -> None:
    ncu = _ncu_result(["uncoalesced_access"], {"sectors": 9.0})
    result = reconcile_evidence(ncu=ncu)
    # uncoalesced_access is claimed; only remaining labels go to unreconciled
    for label in result["unreconciled"].get("ncu", []):
        assert label != "uncoalesced_access"


# ── No false positives ────────────────────────────────────────────────────────

def test_no_diagnoses_when_no_signals() -> None:
    result = reconcile_evidence(
        static=_static([]),
        ncu=_ncu_result([]),
    )
    assert result["diagnoses"] == []


def test_profiler_only_no_sync_gives_no_excessive_sync() -> None:
    result = reconcile_evidence(profiler=_profiler_result(["input_bound"]))
    labels = {d["label"] for d in result["diagnoses"]}
    assert "excessive_synchronization" not in labels


# ── API endpoint ──────────────────────────────────────────────────────────────

def test_api_reconcile_endpoint() -> None:
    import pathlib
    pytest = __import__("pytest")
    pytest.importorskip("fastapi", reason="fastapi not installed")
    pytest.importorskip("httpx", reason="httpx not installed")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from fastapi.testclient import TestClient
    from api import app

    client = TestClient(app)
    response = client.post("/reconcile", json={
        "static": None,
        "ptx": None,
        "ncu": None,
        "profiler": None,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == "reconciliation_v1"
    assert body["diagnoses"] == []


# ── Missing evidence ──────────────────────────────────────────────────────────

def test_missing_evidence_present_when_ncu_not_confirming() -> None:
    # Source confirms inefficient_global_memory_access; NCU not provided
    result = reconcile_evidence(static=_static(["strided_or_pitched"], ["strided_or_pitched"]))
    d = next(d for d in result["diagnoses"] if d["label"] == "inefficient_global_memory_access")
    assert d["missing_evidence"] is not None


def test_missing_evidence_none_when_all_confirming() -> None:
    # Source + NCU both confirm excessive_synchronization + profiler → "confirmed"
    ncu = _ncu_result(["warp_stall_sync"], {"stall": "barrier"})
    result = reconcile_evidence(
        static=_static(["unnecessary_syncthreads"]),
        ncu=ncu,
        profiler=_profiler_result(["sync_bound"]),
    )
    d = next(d for d in result["diagnoses"] if d["label"] == "excessive_synchronization")
    # All 3 layers confirm, so missing_evidence should be None
    assert d["missing_evidence"] is None


def test_missing_evidence_contains_ncu_command() -> None:
    result = reconcile_evidence(static=_static(["strided_or_pitched"], ["strided_or_pitched"]))
    d = next(d for d in result["diagnoses"] if d["label"] == "inefficient_global_memory_access")
    me = d["missing_evidence"]
    assert me["ncu_command"] is not None
    assert me["ncu_command"].startswith("ncu --metrics ")


def test_missing_evidence_full_collection_command() -> None:
    result = reconcile_evidence(static=_static(["strided_or_pitched"], ["strided_or_pitched"]))
    d = next(d for d in result["diagnoses"] if d["label"] == "inefficient_global_memory_access")
    assert d["missing_evidence"]["full_collection_command"] == (
        "ncu --set full --csv ./report.csv ./your_kernel"
    )


def test_missing_evidence_confidence_if_confirmed_upgrades() -> None:
    # With source only (1/2 available), confidence is "low-medium"
    result = reconcile_evidence(
        static=_static(["strided_or_pitched"], ["strided_or_pitched"]),
        ncu=_ncu_result([]),  # ncu present but not confirming
    )
    d = next(d for d in result["diagnoses"] if d["label"] == "inefficient_global_memory_access")
    assert d["confidence"] == "low-medium"
    # If NCU confirms, confidence would upgrade
    conf_after = d["missing_evidence"]["confidence_if_confirmed"]
    assert conf_after in ("high", "medium-high", "confirmed")


def test_missing_evidence_metric_names_for_tensor_core() -> None:
    ncu = _ncu_result(["tensor_core_underutilized"], {"tc": 5.0})
    result = reconcile_evidence(ncu=ncu)
    d = next(d for d in result["diagnoses"] if d["label"] == "tensor_core_underutilization")
    me = d["missing_evidence"]
    # NCU already confirmed but missing_evidence checks evidence_needed for non-confirming layers
    # tensor_core_underutilization has only ncu_check, so if ncu confirms → missing_evidence is None
    assert me is None


def test_missing_evidence_metrics_for_inefficient_global_access() -> None:
    # Source confirms; NCU not provided → NCU metrics should be listed
    result = reconcile_evidence(static=_static(["strided_or_pitched"], ["strided_or_pitched"]))
    d = next(d for d in result["diagnoses"] if d["label"] == "inefficient_global_memory_access")
    metric_names = [m["metric"] for m in d["missing_evidence"]["metrics"]]
    assert any("l1tex" in name for name in metric_names)
    assert any("dram" in name for name in metric_names)


def test_missing_evidence_metrics_for_register_pressure() -> None:
    # PTX confirms register_pressure; NCU not provided
    result = reconcile_evidence(ptx=_ptx_result(_PTX_SPILL))
    d = next(d for d in result["diagnoses"] if d["label"] == "register_pressure")
    metric_names = [m["metric"] for m in d["missing_evidence"]["metrics"]]
    assert any("warps_active" in name for name in metric_names)
    assert any("registers_per_thread" in name for name in metric_names)


def test_what_evidence_is_missing_returns_only_actionable() -> None:
    # inefficient_global_memory_access fires (source), tensor_core does not fire
    result = what_evidence_is_missing(
        static=_static(["strided_or_pitched"], ["strided_or_pitched"])
    )
    labels = {d["label"] for d in result}
    assert "inefficient_global_memory_access" in labels
    # All returned diagnoses must have non-None missing_evidence
    for d in result:
        assert d["missing_evidence"] is not None


def test_what_evidence_is_missing_empty_when_all_confirmed() -> None:
    ncu = _ncu_result(["warp_stall_sync"], {"stall": "barrier"})
    result = what_evidence_is_missing(
        static=_static(["unnecessary_syncthreads"]),
        ncu=ncu,
        profiler=_profiler_result(["sync_bound"]),
    )
    # excessive_synchronization is confirmed by 3 layers → not returned
    labels = {d["label"] for d in result}
    assert "excessive_synchronization" not in labels


def test_what_evidence_is_missing_with_ncu_absent() -> None:
    # No NCU provided; warp_divergence fires via PTX
    result = what_evidence_is_missing(ptx=_ptx_result(_PTX_BRANCH))
    labels = {d["label"] for d in result}
    assert "warp_divergence_risk" in labels
    d = next(d for d in result if d["label"] == "warp_divergence_risk")
    metric_names = [m["metric"] for m in d["missing_evidence"]["metrics"]]
    assert any("thread_inst_executed" in name for name in metric_names)


def test_missing_evidence_each_metric_has_required_fields() -> None:
    result = reconcile_evidence(static=_static(["strided_or_pitched"], ["strided_or_pitched"]))
    d = next(d for d in result["diagnoses"] if d["label"] == "inefficient_global_memory_access")
    for m in d["missing_evidence"]["metrics"]:
        assert "metric" in m
        assert "label" in m
        assert "why" in m
        assert "layer" in m


# ── Helpers imported at module level to avoid name confusion ──────────────────

try:
    from pytest import approx as pytest_approx  # type: ignore[attr-defined]
except ImportError:
    def pytest_approx(x, **_):  # type: ignore[misc]
        return x
