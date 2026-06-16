from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex.thresholds import (
    CLASSIFIER_VERSION,
    ClassifierThresholds,
    DEFAULT_THRESHOLDS,
    resolve_thresholds,
)
from fournex.ncu_analysis import classify_ncu_bottlenecks, analyze_ncu_csv_text
from fournex.analysis import classify_bottlenecks, summarize_run, summarize_run_with_steady_state


# ---------------------------------------------------------------------------
# Regression contract: default values must equal the historical literals
# ---------------------------------------------------------------------------

def test_defaults_match_historical_literals():
    t = DEFAULT_THRESHOLDS
    assert t.ncu_dram_throughput_high_pct == 70.0
    assert t.ncu_memory_stall_fraction_min == 0.50
    assert t.ncu_dominant_warp_stall_pct == 20.0
    assert t.ncu_l1_hit_low_pct == 40.0
    assert t.ncu_l2_hit_low_pct == 50.0
    assert t.ncu_load_sectors_per_request_high == 4.0
    assert t.ncu_tc_util_low_pct == 30.0
    assert t.ncu_tc_occupancy_ok_pct == 40.0
    assert t.ncu_occupancy_low_pct == 40.0
    assert t.ncu_eligible_warps_low == 1.0
    assert t.ncu_scheduler_active_low_pct == 40.0
    assert t.ncu_issue_slot_low_pct == 60.0
    assert t.input_bound_ratio == 0.2
    assert t.copy_bound_ratio == 0.15
    assert t.sync_bound_ratio == 0.1
    assert t.underutilized_gpu_util_pct == 35.0
    assert t.memory_pressure_peak_ratio == 0.9
    assert t.shape_volatility_ratio == 0.3
    assert t.launch_bound_gpu_util_pct == 50.0
    assert t.launch_bound_stall_ratio_max == 0.1
    assert t.fat_high_threshold == 45.0
    assert t.fat_moderate_threshold == 20.0


# ---------------------------------------------------------------------------
# resolve_thresholds
# ---------------------------------------------------------------------------

def test_resolve_none_env_returns_defaults():
    resolved = resolve_thresholds(None)
    assert resolved.values == DEFAULT_THRESHOLDS
    assert resolved.source == "defaults"
    assert resolved.sm_version is None
    assert len(resolved.thresholds_hash) == 12


def test_resolve_empty_env_returns_defaults():
    resolved = resolve_thresholds({})
    assert resolved.values == DEFAULT_THRESHOLDS
    assert resolved.source == "defaults"


def test_resolve_deterministic_hash():
    h1 = resolve_thresholds(None).thresholds_hash
    h2 = resolve_thresholds(None).thresholds_hash
    assert h1 == h2


def test_resolve_arch_override_changes_only_targeted_field():
    env = {
        "gpu_model": "h100",
        "arch_profile_overrides": {
            "profiles": {
                "sm_90": {"classifier_thresholds": {"ncu_issue_slot_low_pct": 50.0}}
            }
        },
    }
    resolved = resolve_thresholds(env)
    assert resolved.values.ncu_issue_slot_low_pct == 50.0
    assert resolved.values.ncu_dram_throughput_high_pct == 70.0  # unchanged
    assert resolved.source == "defaults+arch_overrides"
    assert resolved.thresholds_hash != resolve_thresholds(None).thresholds_hash


def test_resolve_unknown_override_key_raises():
    env = {
        "gpu_model": "h100",
        "arch_profile_overrides": {
            "profiles": {
                "sm_90": {"classifier_thresholds": {"nonexistent_field": 99.0}}
            }
        },
    }
    with pytest.raises(ValueError, match="nonexistent_field"):
        resolve_thresholds(env)


# ---------------------------------------------------------------------------
# classify_ncu_bottlenecks — with/without explicit defaults are equivalent
# ---------------------------------------------------------------------------

def _ncu_summary_with_issue_slot(isu_pct: float) -> dict:
    return {
        "kernels_with_ncu_data": 1,
        "kernel_count": 1,
        "avg_dram_throughput_pct": 50.0,
        "avg_tensor_core_utilization_pct": 80.0,
        "avg_l1_cache_hit_rate_pct": 70.0,
        "avg_l2_cache_hit_rate_pct": 80.0,
        "avg_global_load_sectors_per_request": 1.0,
        "avg_issue_slot_utilization_pct": isu_pct,
        "avg_occupancy_pct": 80.0,
        "avg_eligible_warps_per_scheduler": 2.0,
        "avg_scheduler_active_pct": 80.0,
        "memory_stall_fraction": 0.1,
        "compute_stall_fraction": 0.1,
        "dominant_warp_stall": "not_selected",
        "dominant_warp_stall_pct": 5.0,
        "warp_stall_breakdown": {},
        "occupancy_limit_causes": [],
    }


def test_classify_ncu_with_defaults_equals_explicit_defaults():
    summary = _ncu_summary_with_issue_slot(55.0)
    result_implicit = classify_ncu_bottlenecks(summary)
    result_explicit = classify_ncu_bottlenecks(summary, thresholds=DEFAULT_THRESHOLDS)
    assert result_implicit == result_explicit


def test_override_flips_classification():
    summary = _ncu_summary_with_issue_slot(55.0)
    # With defaults (60.0 cutoff), 55 triggers low_issue_efficiency
    default_labels = [b["label"] for b in classify_ncu_bottlenecks(summary)]
    assert "low_issue_efficiency" in default_labels

    # With lowered cutoff (50.0), 55 no longer triggers the rule
    lower_cutoff = ClassifierThresholds(ncu_issue_slot_low_pct=50.0)
    overridden_labels = [b["label"] for b in classify_ncu_bottlenecks(summary, thresholds=lower_cutoff)]
    assert "low_issue_efficiency" not in overridden_labels


# ---------------------------------------------------------------------------
# Provenance stamping in result dicts
# ---------------------------------------------------------------------------

def _minimal_ncu_csv() -> str:
    return (
        '"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","123","test","host","vecAdd(float*, float*, float*, int)","2024-Jan-01 00:00:00","1","7","GPU Speed Of Light Throughput","sm__throughput.avg.pct_of_peak_sustained_elapsed","%" ,"42"\n'
    )


def test_ncu_result_has_classifier_block():
    result = analyze_ncu_csv_text(_minimal_ncu_csv())
    assert "classifier" in result
    clf = result["classifier"]
    assert clf["classifier_version"] == CLASSIFIER_VERSION
    assert "thresholds_hash" in clf
    assert "thresholds_source" in clf
    assert len(clf["thresholds_hash"]) == 12


def test_ncu_classifier_version_unchanged():
    result = analyze_ncu_csv_text(_minimal_ncu_csv())
    assert result["classifier"]["classifier_version"] == "0.2.0"


def test_classifier_version_constant():
    assert CLASSIFIER_VERSION == "0.2.0"


# ---------------------------------------------------------------------------
# Provenance in telemetry diagnosis
# ---------------------------------------------------------------------------

def _minimal_events():
    return []


def test_diagnosis_has_thresholds_hash():
    summary = summarize_run(_minimal_events())
    diag = summary.get("diagnosis", {})
    assert "thresholds_hash" in diag
    assert "thresholds_source" in diag
    assert diag["classifier_version"] == "0.2.0"


def _three_step_input_fraction_events(wait_ns: int = 150) -> list[dict]:
    events: list[dict] = [
        {
            "event_type": "gpu_sample",
            "payload": {
                "utilization_gpu_pct": 80,
                "utilization_mem_pct": 30,
                "memory_used_bytes": 30,
                "memory_total_bytes": 100,
            },
        }
    ]
    for step_id in range(1, 4):
        events.extend([
            {"event_type": "step_start", "step_id": step_id, "payload": {"step_kind": "train"}},
            {
                "event_type": "dataloader_span",
                "step_id": step_id,
                "duration_ns": wait_ns,
                "payload": {"stage": "next"},
            },
            {
                "event_type": "phase_span",
                "step_id": step_id,
                "duration_ns": 500,
                "payload": {"phase_name": "forward"},
            },
            {
                "event_type": "step_end",
                "step_id": step_id,
                "duration_ns": 1000,
                "payload": {"step_kind": "train", "status": "ok"},
            },
        ])
    return events


def test_run_with_steady_state_passes_environment_to_both_scopes():
    env = {
        "gpu_model": "h100",
        "arch_profile_overrides": {
            "profiles": {
                "sm_90": {"classifier_thresholds": {"input_bound_ratio": 0.1}}
            }
        },
    }

    default_summary = summarize_run_with_steady_state(_three_step_input_fraction_events())
    overridden_summary = summarize_run_with_steady_state(
        _three_step_input_fraction_events(),
        environment=env,
    )

    assert default_summary["run"]["diagnosis"]["primary_bottleneck"] is None
    assert default_summary["steady_state"]["diagnosis"]["primary_bottleneck"] is None
    assert overridden_summary["run"]["diagnosis"]["primary_bottleneck"] == "input_bound"
    assert overridden_summary["steady_state"]["diagnosis"]["primary_bottleneck"] == "input_bound"
    assert overridden_summary["run"]["diagnosis"]["thresholds_source"] == "defaults+arch_overrides"
    assert overridden_summary["steady_state"]["diagnosis"]["thresholds_source"] == "defaults+arch_overrides"
    assert overridden_summary["run"]["diagnosis"]["thresholds_hash"] != resolve_thresholds(None).thresholds_hash
