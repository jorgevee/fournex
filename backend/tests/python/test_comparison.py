import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as at
from fournex.comparison import compare_implementations


# ── PTX fixtures ──────────────────────────────────────────────────────────────

PTX_HEAVY = """
.version 8.0
.target sm_80
.visible .entry heavy_kernel() {
    .reg .f32 %f<130>;
    .reg .b64 %SP;
    .reg .b64 %SPL;
    .local .align 4 .b8 __local_depot0[512];
    ld.local.f32 %f0, [%SP+0];
    ld.local.f32 %f1, [%SP+4];
    st.local.f32 [%SP+8], %f1;
    ld.global.f32 %f2, [%SP+0];
    ret;
}
"""

PTX_LIGHT = """
.version 8.0
.target sm_80
.visible .entry light_kernel() {
    .reg .f32 %f<32>;
    .reg .b64 %rd<8>;
    ld.global.f32 %f0, [%rd0];
    fma.rn.f32 %f1, %f0, %f0, %f0;
    st.global.f32 [%rd1], %f1;
    ret;
}
"""


# ── NCU CSV fixtures ──────────────────────────────────────────────────────────

def _ncu_csv(*rows: str) -> str:
    return "\n".join(["Kernel Name,Metric Name,Metric Unit,Metric Value"] + list(rows))


NCU_BAD = _ncu_csv(
    "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,90.0",
    "ker,l1tex__t_sector_hit_rate.pct,%,20.0",
    "ker,lts__t_sector_hit_rate.pct,%,30.0",
    "ker,sm__issue_active.avg.pct_of_peak_sustained_active,%,25.0",
)

NCU_GOOD = _ncu_csv(
    "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,40.0",
    "ker,l1tex__t_sector_hit_rate.pct,%,75.0",
    "ker,lts__t_sector_hit_rate.pct,%,85.0",
    "ker,sm__issue_active.avg.pct_of_peak_sustained_active,%,70.0",
)


def _side(label: str, **kwargs) -> dict:
    return {"label": label, "cuda_filename": "<m>", "ptx_filename": "k.ptx", **kwargs}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_ptx_register_diff_reflects_reduction() -> None:
    result = compare_implementations(
        _side("heavy", ptx=PTX_HEAVY),
        _side("light", ptx=PTX_LIGHT),
    )
    rd = result["ptx_diff"]["register_count"]
    assert rd["a"] > rd["b"]
    assert rd["delta"] < 0
    assert rd["better"] == "b"


def test_ptx_spill_vs_no_spill() -> None:
    result = compare_implementations(
        _side("spill",    ptx=PTX_HEAVY),
        _side("no_spill", ptx=PTX_LIGHT),
    )
    spill_diff = result["ptx_diff"]["has_register_spills"]
    assert spill_diff["a"] is True
    assert spill_diff["b"] is False
    assert spill_diff["resolved_in_b"] is True
    assert spill_diff["introduced_in_b"] is False


def test_ptx_instruction_mix_diff_present() -> None:
    result = compare_implementations(
        _side("a", ptx=PTX_HEAVY),
        _side("b", ptx=PTX_LIGHT),
    )
    mix_diff = result["ptx_diff"]["instruction_mix_diff"]
    assert "global_loads" in mix_diff
    assert isinstance(mix_diff["global_loads"]["delta"], int)


def test_ncu_metrics_diff_dram_and_cache() -> None:
    result = compare_implementations(
        _side("bad",  ncu_csv=NCU_BAD),
        _side("good", ncu_csv=NCU_GOOD),
    )
    ncu = result["ncu_diff"]
    assert ncu["available"] is True
    assert ncu["avg_dram_throughput_pct"]["better"] == "b"
    assert ncu["avg_l1_cache_hit_rate_pct"]["better"] == "b"
    assert ncu["avg_l2_cache_hit_rate_pct"]["better"] == "b"


def test_ptx_findings_resolved_and_new() -> None:
    result = compare_implementations(
        _side("a", ptx=PTX_HEAVY),
        _side("b", ptx=PTX_LIGHT),
    )
    fd = result["ptx_diff"]["findings_diff"]
    assert "register_spills_detected" in fd["resolved_in_b"]
    assert "very_high_register_count" in fd["resolved_in_b"]
    assert isinstance(fd["new_in_b"], list)
    assert isinstance(fd["shared"], list)


def test_full_scorecard_b_wins() -> None:
    result = compare_implementations(
        _side("baseline",  ptx=PTX_HEAVY, ncu_csv=NCU_BAD),
        _side("optimized", ptx=PTX_LIGHT, ncu_csv=NCU_GOOD),
    )
    verdict = result["verdict"]
    assert verdict["overall_winner"] == "b"
    assert verdict["score_b"] > verdict["score_a"]
    assert verdict["score_delta"] > 0
    assert "register_efficiency" in verdict["dimensions_won_by_b"]


def test_identical_inputs_produce_tie() -> None:
    result = compare_implementations(
        _side("x", ptx=PTX_LIGHT),
        _side("y", ptx=PTX_LIGHT),
    )
    verdict = result["verdict"]
    assert verdict["overall_winner"] == "tie"
    assert verdict["score_a"] == verdict["score_b"]
    assert verdict["score_delta"] == 0.0


def test_partial_data_ncu_only_graceful_skip() -> None:
    result = compare_implementations(
        _side("a", ncu_csv=NCU_BAD),
        _side("b", ncu_csv=NCU_GOOD),
    )
    assert result["ptx_diff"]["available"] is False
    assert result["static_diff"]["available"] is False
    assert result["ncu_diff"]["available"] is True
    scorecard = result["scorecard"]
    assert scorecard["register_efficiency"]["available"] is False
    assert scorecard["memory_efficiency"]["available"] is True


def test_response_schema_and_required_fields() -> None:
    result = compare_implementations(
        _side("a", ptx=PTX_LIGHT),
        _side("b", ptx=PTX_LIGHT),
    )
    assert result["schema"] == "comparison_v1"
    for key in ("label_a", "label_b", "data_availability", "static_diff",
                "ptx_diff", "ncu_diff", "scorecard", "verdict"):
        assert key in result
    for dim in ("register_efficiency", "memory_efficiency",
                "compute_efficiency", "launch_efficiency"):
        assert dim in result["scorecard"]


def test_api_compare_endpoint() -> None:
    import pathlib
    pytest = __import__("pytest")
    fastapi = pytest.importorskip("fastapi", reason="fastapi not installed in this environment")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from fastapi.testclient import TestClient
    from api import app

    client = TestClient(app)
    response = client.post("/compare", json={
        "a": {"label": "baseline",  "ptx": PTX_LIGHT, "cuda_filename": "<m>", "ptx_filename": "l.ptx"},
        "b": {"label": "optimized", "ptx": PTX_HEAVY, "cuda_filename": "<m>", "ptx_filename": "h.ptx"},
    })
    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == "comparison_v1"
    assert body["label_a"] == "baseline"
    assert body["label_b"] == "optimized"


def test_labels_propagated_to_response() -> None:
    result = compare_implementations(
        _side("kernel_v1", ptx=PTX_LIGHT),
        _side("kernel_v2", ptx=PTX_HEAVY),
    )
    assert result["label_a"] == "kernel_v1"
    assert result["label_b"] == "kernel_v2"


def test_finding_codes_are_strings_in_diff() -> None:
    result = compare_implementations(
        _side("a", ptx=PTX_HEAVY),
        _side("b", ptx=PTX_LIGHT),
    )
    fd = result["ptx_diff"]["findings_diff"]
    for key in ("resolved_in_b", "new_in_b", "shared"):
        assert isinstance(fd[key], list)
        for code in fd[key]:
            assert isinstance(code, str)
            assert len(code) > 0
