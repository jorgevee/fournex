"""Tests for the `frx eval sakana` harness and the Sakana NCU adapter.

These run fully offline against the packaged cached subset (no GPU, no network).
The adapter tests pin the one trap that motivated a dedicated adapter: the
byte/second "Memory Throughput" metric must never be coerced into a %-of-peak
field.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as at
from fournex import eval_sakana as ev
from fournex.sakana_ncu_adapter import ABSENT_SECTIONS, adapt_ncu_profile, parse_ncu_profile

SUBSET = ev.DEFAULT_SUBSET
HAS_SUBSET = SUBSET.exists()


def _confidence_le(label, ceiling):
    order = ev._CONFIDENCE_ORDER
    return label is None or order.index(label) <= order.index(ceiling)


# A realistic Sakana NCU_Profile as a Python-repr string (single quotes,
# apostrophe inside a rule description, byte/second Memory Throughput).
SAMPLE_PROFILE = (
    "{'metrics': {"
    "'Issue Slots Busy': {'unit': '%', 'avg_value': 42.7, 'variance': 0.0, 'n': 5}, "
    "'SM Busy': {'unit': '%', 'avg_value': 42.7, 'variance': 0.0, 'n': 5}, "
    "'Memory Throughput': {'unit': 'byte/second', 'avg_value': 21164232027.7, 'variance': 0.0, 'n': 5}, "
    "'Mem Busy': {'unit': '%', 'avg_value': 89.9, 'variance': 0.0, 'n': 5}, "
    "'Max Bandwidth': {'unit': '%', 'avg_value': 85.6, 'variance': 0.0, 'n': 5}, "
    "'L1/TEX Hit Rate': {'unit': '%', 'avg_value': 12.0, 'variance': 0.0, 'n': 5}, "
    "'L2 Hit Rate': {'unit': '%', 'avg_value': 97.0, 'variance': 0.0, 'n': 5}, "
    "'Block Limit Registers': {'unit': 'block', 'avg_value': 2.0, 'variance': 0.0, 'n': 5}, "
    "'Block Limit Shared Mem': {'unit': 'block', 'avg_value': 3.0, 'variance': 0.0, 'n': 5}, "
    "'Block Limit Warps': {'unit': 'block', 'avg_value': 8.0, 'variance': 0.0, 'n': 5}, "
    "'Theoretical Occupancy': {'unit': '%', 'avg_value': 100.0, 'variance': 0.0, 'n': 5}, "
    "'Achieved Occupancy': {'unit': '%', 'avg_value': 25.0, 'variance': 0.0, 'n': 5}"
    "}, 'rules': {'Occupancy': {'type': 'INF', 'description': \"this kernel's occupancy is fine\"}}}"
)


# ── Adapter ───────────────────────────────────────────────────────────────────

def test_parse_handles_python_repr_with_apostrophes():
    parsed = parse_ncu_profile(SAMPLE_PROFILE)
    assert parsed is not None
    assert "metrics" in parsed and "rules" in parsed


def test_parse_rejects_garbage_and_null():
    assert parse_ncu_profile(None) is None
    assert parse_ncu_profile("") is None
    assert parse_ncu_profile("not a dict") is None
    assert parse_ncu_profile("{'metrics':") is None  # truncated/invalid


def test_byte_per_second_memory_throughput_not_treated_as_percent():
    summary = adapt_ncu_profile(SAMPLE_PROFILE)
    assert summary is not None
    # Max Bandwidth (%) is the only DRAM %-of-peak figure; the byte/second
    # "Memory Throughput" must NOT leak into dram_throughput_pct.
    assert summary.dram_throughput_pct == pytest.approx(85.6)
    assert summary.dram_throughput_pct <= 100.0
    # The raw byte/second value is preserved for transparency, in metrics only.
    assert summary.metrics.get("memory_throughput_bytes_per_s") == pytest.approx(21164232027.7)


def test_percent_fields_mapped_correctly():
    s = adapt_ncu_profile(SAMPLE_PROFILE)
    assert s.memory_busy_pct == pytest.approx(89.9)
    assert s.sm_throughput_pct == pytest.approx(42.7)
    assert s.issue_slot_utilization_pct == pytest.approx(42.7)
    assert s.l1_cache_hit_rate_pct == pytest.approx(12.0)
    assert s.l2_cache_hit_rate_pct == pytest.approx(97.0)
    assert s.achieved_occupancy_pct == pytest.approx(25.0)
    assert s.theoretical_occupancy_pct == pytest.approx(100.0)


def test_occupancy_limiting_factor_from_lowest_block_limit():
    # Registers=2 is the binding (minimum) block limit -> "registers".
    s = adapt_ncu_profile(SAMPLE_PROFILE)
    assert s.occupancy_estimate["limiting_factors"] == ["registers"]


def test_absent_sections_surface_as_missing_evidence():
    # The sample (like the whole dataset) lacks stall, coalescing and tensor data.
    s = adapt_ncu_profile(SAMPLE_PROFILE)
    assert s.warp_stall_breakdown == {}
    assert s.global_load_sectors_per_request is None
    assert s.tensor_core_utilization_pct is None
    assert set(ABSENT_SECTIONS) >= {
        "warp_stall_breakdown", "global_load_sectors_per_request", "tensor_core_utilization_pct"
    }


def test_adapter_returns_none_without_metrics():
    assert adapt_ncu_profile("{'rules': {}}") is None
    assert adapt_ncu_profile(None) is None


def test_low_occupancy_profile_drives_occupancy_diagnosis():
    ncu = at.analyze_ncu_profile_dict(SAMPLE_PROFILE)
    assert ncu is not None
    labels = {b["label"] for b in ncu["bottlenecks"]}
    assert "occupancy_limited" in labels  # achieved 25% < threshold
    assert "occupancy_limited_by_registers" in labels


# ── Harness (requires the cached subset) ──────────────────────────────────────

@pytest.mark.skipif(not HAS_SUBSET, reason="cached Sakana subset not present")
def test_evaluate_row_shape_and_no_gpu():
    rows = ev.load_rows()
    rec = ev.evaluate_row(rows[0])
    for key in ("row_key", "primary_diagnosis", "confidence", "diagnoses",
                "missing_evidence_metrics", "absent_ncu_sections", "weak_label",
                "correctness", "summary_headline", "ground_truth"):
        assert key in rec
    assert rec["correctness"]["status"] == "not_verified_by_fournex"


@pytest.mark.skipif(not HAS_SUBSET, reason="cached Sakana subset not present")
def test_run_eval_full_subset_runs_clean():
    res = ev.run_eval(use_gold=False)
    lb = res["leaderboard"]
    assert res["schema"] == "sakana_eval_v1"
    assert lb["rows_evaluated"] == len(res["per_row"]) > 0
    assert lb["no_crash_rate"]["value"] == 1.0
    # Structural over-claim guard: this dataset can never confirm a diagnosis at
    # better than medium-high (no warp-stall data), and in practice tops out at medium.
    assert lb["confidence_ceiling_respected"]["value"] == 1.0
    assert lb["confidence_ceiling_respected"]["ceiling"] == "medium-high"
    assert _confidence_le(lb["max_confidence_emitted"]["value"], "medium-high")
    # Correctness is never inferred from the profile.
    assert "documented blind spot" in lb["correctness_warning_recall"]["silent_numerical_mismatch"]["note"]


@pytest.mark.skipif(not HAS_SUBSET, reason="cached Sakana subset not present")
def test_correctness_warning_fires_on_build_errors_only():
    res = ev.run_eval(use_gold=False)
    cwr = res["leaderboard"]["correctness_warning_recall"]
    # Every build/runtime failure (Error text) is flagged; silent mismatches are not.
    assert cwr["build_or_runtime_error"]["value"] == 1.0
    assert cwr["silent_numerical_mismatch"]["value"] == 0.0


@pytest.mark.skipif(not HAS_SUBSET, reason="cached Sakana subset not present")
def test_gold_keys_resolve_and_pass():
    res = ev.run_eval(use_gold=True)
    gold = res["leaderboard"]["gold"]
    # Every hand-labeled row must resolve in the cached subset (no stale keys).
    assert gold["gold_rows_in_subset"] == gold["gold_rows_total"]
    assert gold["primary_bottleneck_accuracy"]["value"] == 1.0
    assert gold["confidence_ceiling_respected"]["value"] == 1.0
    assert gold["correctness_warning_accuracy"]["value"] == 1.0
    assert gold["failures"] == []


@pytest.mark.skipif(not HAS_SUBSET, reason="cached Sakana subset not present")
def test_sampling_is_deterministic():
    a = ev.load_rows(sample=10, seed=7)
    b = ev.load_rows(sample=10, seed=7)
    c = ev.load_rows(sample=10, seed=8)
    assert [ev.row_key(r) for r in a] == [ev.row_key(r) for r in b]
    assert [ev.row_key(r) for r in a] != [ev.row_key(r) for r in c]


@pytest.mark.skipif(not HAS_SUBSET, reason="cached Sakana subset not present")
def test_level_filter():
    res = ev.run_eval(level=1, use_gold=False)
    assert all(r["level"] == 1 for r in res["per_row"])
    assert res["per_row"]
