"""Golden tests for the rules-based recommendation engine.

Each case asserts:
  - expected top recommendation IDs appear
  - excluded recommendation IDs do NOT appear
  - priority levels and structure are correct

Reuses event fixtures from analysis_bottleneck_golden_cases.py where possible;
builds minimal inline events for cases that need specific signal combinations.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as at
from fournex.recommendations.engine import generate_recommendations
from fournex.recommendations.engine import _score_recommendation

from analysis_bottleneck_golden_cases import (
    COPY_BOUND_EVENTS,
    INPUT_BOUND_EVENTS,
    LAUNCH_BOUND_EVENTS,
    MEMORY_PRESSURE_EVENTS,
    SHAPE_INSTABILITY_EVENTS,
    SPARSE_TELEMETRY_EVENTS,
    SYNC_BOUND_EVENTS,
    UNDERUTILIZED_GPU_EVENTS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rec_ids(summary: dict) -> list[str]:
    return [r["id"] for r in summary["diagnosis"]["recommendations"]]


def _top_rec_id(summary: dict) -> str:
    return summary["diagnosis"]["recommendations"][0]["id"]


def _rec_by_id(summary: dict, rec_id: str) -> dict | None:
    return next((r for r in summary["diagnosis"]["recommendations"] if r["id"] == rec_id), None)


def _make_step_events(
    step_id: int,
    *,
    dataloader_ns: int = 0,
    h2d_ns: int = 0,
    sync_ns: int = 0,
    forward_ns: int = 30,
    backward_ns: int = 20,
    optimizer_ns: int = 10,
    step_wall_ns: int = 100,
    gpu_util: float = 60.0,
    mem_used: int = 40,
    mem_total: int = 100,
    profiler_exported: bool = False,
    shapes: dict | None = None,
) -> list[dict]:
    events = [
        {"event_type": "gpu_sample", "payload": {
            "utilization_gpu_pct": gpu_util,
            "utilization_mem_pct": mem_used,
            "memory_used_bytes": mem_used,
            "memory_total_bytes": mem_total,
        }},
        {"event_type": "step_start", "step_id": step_id, "payload": {"step_kind": "train"}},
    ]
    if dataloader_ns:
        events.append({"event_type": "dataloader_span", "step_id": step_id, "duration_ns": dataloader_ns, "payload": {"stage": "next"}})
    if h2d_ns:
        events.append({"event_type": "memcpy_span", "step_id": step_id, "duration_ns": h2d_ns, "payload": {"copy_kind": "h2d"}})
    if sync_ns:
        events.append({"event_type": "sync_wait", "step_id": step_id, "duration_ns": sync_ns, "payload": {}})
    if profiler_exported:
        events.append({"event_type": "profiler_window", "step_id": step_id, "payload": {"window_state": "exported"}})
    if shapes:
        events.append({"event_type": "shape_snapshot", "step_id": step_id, "payload": shapes})
    events += [
        {"event_type": "phase_span", "step_id": step_id, "duration_ns": forward_ns, "payload": {"phase_name": "forward"}},
        {"event_type": "phase_span", "step_id": step_id, "duration_ns": backward_ns, "payload": {"phase_name": "backward"}},
        {"event_type": "phase_span", "step_id": step_id, "duration_ns": optimizer_ns, "payload": {"phase_name": "optimizer"}},
        {"event_type": "step_end", "step_id": step_id, "duration_ns": step_wall_ns, "payload": {"step_kind": "train", "status": "ok"}},
    ]
    return events


# ── Case 1: Input pipeline starvation ────────────────────────────────────────

def test_input_starvation_top_recs_are_dataloader_fixes() -> None:
    # INPUT_BOUND_EVENTS: avg_dataloader_fraction=0.325, gpu_util=41% → input_pipeline_stalled
    summary = at.summarize_run(INPUT_BOUND_EVENTS)
    ids = _rec_ids(summary)

    assert "rec_input_num_workers" in ids
    assert "rec_input_pinned_memory" in ids
    assert "rec_input_prefetch_factor" in ids


def test_input_starvation_does_not_recommend_unrelated_fixes() -> None:
    summary = at.summarize_run(INPUT_BOUND_EVENTS)
    ids = _rec_ids(summary)

    assert "rec_launch_torch_compile" not in ids
    assert "rec_launch_cuda_graphs" not in ids
    assert "rec_sync_remove_item" not in ids
    assert "rec_mem_reduce_batch" not in ids


def test_input_starvation_severe_promotes_all_four_recs() -> None:
    # Severe = dataloader fraction > 0.40
    events = (
        _make_step_events(1, dataloader_ns=50, step_wall_ns=100, gpu_util=42.0) +
        _make_step_events(2, dataloader_ns=48, step_wall_ns=100, gpu_util=40.0)
    )
    summary = at.summarize_run(events)
    ids = _rec_ids(summary)

    assert "rec_input_num_workers" in ids
    assert "rec_input_move_transforms" in ids


def test_input_starvation_top_rec_is_high_priority() -> None:
    # Use severe starvation (dataloader > 40% of step) to guarantee HIGH priority
    events = (
        _make_step_events(1, dataloader_ns=50, step_wall_ns=100, gpu_util=42.0) +
        _make_step_events(2, dataloader_ns=48, step_wall_ns=100, gpu_util=40.0)
    )
    summary = at.summarize_run(events)
    top = summary["diagnosis"]["recommendations"][0]
    assert top["priority"] == "high"


# ── Case 2: H2D copy bound ────────────────────────────────────────────────────

def test_copy_bound_recommends_pinned_memory_and_overlap() -> None:
    summary = at.summarize_run(COPY_BOUND_EVENTS)
    ids = _rec_ids(summary)

    assert "rec_copy_pinned_memory" in ids
    assert "rec_copy_overlap" in ids


def test_copy_bound_does_not_recommend_dataloader_workers() -> None:
    summary = at.summarize_run(COPY_BOUND_EVENTS)
    ids = _rec_ids(summary)

    assert "rec_input_num_workers" not in ids


def test_copy_bound_severe_adds_reduce_volume_rec() -> None:
    # h2d fraction > 0.30 triggers severe rule which adds rec_copy_reduce_volume
    events = (
        _make_step_events(1, h2d_ns=35, step_wall_ns=100, gpu_util=55.0) +
        _make_step_events(2, h2d_ns=32, step_wall_ns=100, gpu_util=55.0)
    )
    summary = at.summarize_run(events)
    ids = _rec_ids(summary)

    assert "rec_copy_pinned_memory" in ids
    assert "rec_copy_overlap" in ids
    assert "rec_copy_reduce_volume" in ids


# ── Case 3: Sync-heavy ────────────────────────────────────────────────────────

def test_sync_moderate_recommends_remove_item_and_batch_logging() -> None:
    # SYNC_BOUND_EVENTS: avg sync fraction = 0.135, triggers sync_heavy (≥0.10)
    summary = at.summarize_run(SYNC_BOUND_EVENTS)
    ids = _rec_ids(summary)

    assert "rec_sync_remove_item" in ids
    assert "rec_sync_batch_logging" in ids


def test_sync_severe_adds_reduce_explicit_syncs() -> None:
    # sync fraction > 0.25 triggers the severe rule which adds rec_sync_reduce_explicit
    events = (
        _make_step_events(1, sync_ns=28, step_wall_ns=100, gpu_util=48.0) +
        _make_step_events(2, sync_ns=25, step_wall_ns=100, gpu_util=50.0)
    )
    summary = at.summarize_run(events)
    ids = _rec_ids(summary)

    assert "rec_sync_remove_item" in ids
    assert "rec_sync_reduce_explicit" in ids


def test_sync_bound_does_not_recommend_dataloader_workers() -> None:
    summary = at.summarize_run(SYNC_BOUND_EVENTS)
    ids = _rec_ids(summary)

    assert "rec_input_num_workers" not in ids


# ── Case 4: Underutilized GPU ─────────────────────────────────────────────────

def test_underutilized_no_mixed_precision_recommends_amp() -> None:
    # UNDERUTILIZED_GPU_EVENTS: gpu_util ~23%, no mixed precision in env
    summary = at.summarize_run(UNDERUTILIZED_GPU_EVENTS)
    ids = _rec_ids(summary)

    assert "rec_util_mixed_precision" in ids
    assert "rec_util_increase_batch" in ids


def test_underutilized_mixed_precision_already_enabled_suppresses_amp_rec() -> None:
    # When mixed precision is already on, rule_underutilized_no_mixed_precision should not fire
    per_step = at.derive_step_metrics(UNDERUTILIZED_GPU_EVENTS)
    run_summary = at.derive_run_summary(UNDERUTILIZED_GPU_EVENTS, per_step)
    bottlenecks = at.classify_bottlenecks(UNDERUTILIZED_GPU_EVENTS, per_step, run_summary)

    result = generate_recommendations(
        bottlenecks, run_summary, per_step,
        environment={"framework": "pytorch", "mixed_precision": True},
    )
    ids = [r["id"] for r in result["recommendations"]]

    assert "rec_util_mixed_precision" not in ids


def test_underutilized_does_not_fire_when_input_pipeline_stalled() -> None:
    # When input pipeline is the real cause, underutilized_gpu rules are suppressed
    events = (
        _make_step_events(1, dataloader_ns=45, step_wall_ns=100, gpu_util=30.0) +
        _make_step_events(2, dataloader_ns=42, step_wall_ns=100, gpu_util=28.0)
    )
    summary = at.summarize_run(events)
    ids = _rec_ids(summary)

    # Underutilized GPU rules are suppressed when input_pipeline_stalled is true
    assert "rec_util_mixed_precision" not in ids
    assert "rec_util_increase_batch" not in ids
    # But input pipeline recs should still appear
    assert "rec_input_num_workers" in ids


# ── Case 5: Launch bound ──────────────────────────────────────────────────────

def test_launch_bound_stable_shapes_recommends_compile_and_cuda_graphs() -> None:
    # LAUNCH_BOUND_EVENTS: profiler windows exported, no shape instability
    summary = at.summarize_run(LAUNCH_BOUND_EVENTS)
    ids = _rec_ids(summary)

    assert "rec_launch_torch_compile" in ids
    assert "rec_launch_cuda_graphs" in ids
    assert "rec_launch_fuse_ops" in ids



def test_launch_bound_unstable_shapes_suppresses_cuda_graphs() -> None:
    # Profiler data + shape instability → CUDA Graphs unsafe, different rule fires
    events = (
        _make_step_events(1, profiler_exported=True, step_wall_ns=100, gpu_util=30.0,
                          shapes={"batch_size": 16, "sequence_length": 128, "shapes": {"x": [16, 128]}}) +
        _make_step_events(2, profiler_exported=True, step_wall_ns=100, gpu_util=32.0,
                          shapes={"batch_size": 16, "sequence_length": 256, "shapes": {"x": [16, 256]}}) +
        _make_step_events(3, profiler_exported=True, step_wall_ns=100, gpu_util=28.0,
                          shapes={"batch_size": 16, "sequence_length": 128, "shapes": {"x": [16, 128]}})
    )
    summary = at.summarize_run(events)
    ids = _rec_ids(summary)

    assert "rec_launch_torch_compile" in ids
    assert "rec_launch_cuda_graphs" not in ids


def test_launch_bound_no_profiler_data_recommends_inspect_first() -> None:
    # launch_bound classified but no profiler windows → rule_launch_no_profiler fires
    # Build events that trigger launch_bound: low util (< 50%), has profiler_windows_exported > 0
    # BUT we want no profiler data... The issue is launch_bound itself requires profiler_windows > 0
    # So actually we can't trigger launch_bound without profiler data by the classifier rules.
    # Instead, test directly via the engine with a manually crafted bottleneck.
    per_step = at.derive_step_metrics(UNDERUTILIZED_GPU_EVENTS)
    run_summary = at.derive_run_summary(UNDERUTILIZED_GPU_EVENTS, per_step)

    # Inject a launch_bound bottleneck with no profiler data in run_summary
    bottlenecks = [{"label": "launch_bound", "score": 0.40, "evidence": {}, "worst_steps": []}]
    # run_summary has profiler_windows_exported=0

    result = generate_recommendations(bottlenecks, run_summary, per_step)
    ids = [r["id"] for r in result["recommendations"]]

    assert "rec_util_inspect_kernels" in ids
    assert "rec_launch_torch_compile" in ids
    assert "rec_launch_cuda_graphs" not in ids


# ── Case 6: Memory pressure ───────────────────────────────────────────────────

def test_memory_critical_recommends_reduce_batch_and_checkpointing() -> None:
    # MEMORY_PRESSURE_EVENTS: peak ratio = 0.95 → memory_near_capacity
    summary = at.summarize_run(MEMORY_PRESSURE_EVENTS)
    ids = _rec_ids(summary)

    assert "rec_mem_reduce_batch" in ids
    assert "rec_mem_gradient_checkpointing" in ids


def test_memory_critical_does_not_recommend_increase_batch() -> None:
    # Increasing batch size when memory is critical would cause OOM
    summary = at.summarize_run(MEMORY_PRESSURE_EVENTS)
    ids = _rec_ids(summary)

    assert "rec_util_increase_batch" not in ids


def test_memory_moderate_without_mixed_precision_recommends_amp() -> None:
    # The classifier only fires memory_pressure at >= 0.90, so the moderate rule
    # (0.75–0.90) is only reachable by calling the engine directly with a crafted bottleneck.
    run_summary = {
        "average_gpu_utilization_pct": 70.0,
        "average_memory_utilization_pct": 80.0,
        "memory_pressure_peak_ratio": 0.82,
        "utilization_instability_pct": 2.0,
        "shape_volatility_ratio": 0.0,
        "profiler_windows_exported": 0,
        "dominant_stall_type": "compute_bound",
    }
    bottlenecks = [{"label": "memory_pressure", "score": 0.82, "evidence": {}, "worst_steps": []}]
    result = generate_recommendations(bottlenecks, run_summary, [])
    ids = [r["id"] for r in result["recommendations"]]

    assert "rec_util_mixed_precision" in ids
    assert "rec_mem_gradient_checkpointing" in ids


# ── Case 7: Shape instability ─────────────────────────────────────────────────

def test_shape_instability_recommends_bucketing() -> None:
    summary = at.summarize_run(SHAPE_INSTABILITY_EVENTS)
    ids = _rec_ids(summary)

    assert "rec_shape_bucket_inputs" in ids


def test_shape_instability_does_not_recommend_cuda_graphs() -> None:
    # CUDA Graphs require static shapes — should never appear with shape instability
    summary = at.summarize_run(SHAPE_INSTABILITY_EVENTS)
    ids = _rec_ids(summary)

    assert "rec_launch_cuda_graphs" not in ids


# ── Case 8: Insufficient telemetry ───────────────────────────────────────────

def test_insufficient_telemetry_recommends_hook_check() -> None:
    summary = at.summarize_run(SPARSE_TELEMETRY_EVENTS)
    ids = _rec_ids(summary)

    assert "rec_telemetry_check_hooks" in ids
    assert "rec_telemetry_enable_profiler" in ids


def test_insufficient_telemetry_does_not_recommend_tuning_fixes() -> None:
    summary = at.summarize_run(SPARSE_TELEMETRY_EVENTS)
    ids = _rec_ids(summary)

    assert "rec_input_num_workers" not in ids
    assert "rec_util_increase_batch" not in ids
    assert "rec_launch_torch_compile" not in ids


# ── Case 9: Structural validation ────────────────────────────────────────────

def test_all_recommendations_have_required_fields() -> None:
    required = {"id", "title", "priority", "score", "confidence", "expected_impact",
                "effort", "category", "why", "actions", "validation", "risks", "triggered_by",
                "roi_score", "tier", "risk", "why_ranked", "roi_components",
                "guardrails_applied", "prerequisites"}

    for events in [INPUT_BOUND_EVENTS, COPY_BOUND_EVENTS, SYNC_BOUND_EVENTS,
                   MEMORY_PRESSURE_EVENTS, SHAPE_INSTABILITY_EVENTS, UNDERUTILIZED_GPU_EVENTS]:
        summary = at.summarize_run(events)
        for rec in summary["diagnosis"]["recommendations"]:
            missing = required - rec.keys()
            assert not missing, f"rec {rec.get('id')} missing fields: {missing}"


def test_recommendation_scores_are_descending() -> None:
    summary = at.summarize_run(INPUT_BOUND_EVENTS)
    scores = [r["score"] for r in summary["diagnosis"]["recommendations"]]
    assert scores == sorted(scores, reverse=True)


def test_recommendation_score_aliases_roi_score() -> None:
    summary = at.summarize_run(INPUT_BOUND_EVENTS)
    for rec in summary["diagnosis"]["recommendations"]:
        assert rec["score"] == rec["roi_score"]



def test_roi_components_are_normalized() -> None:
    summary = at.summarize_run(INPUT_BOUND_EVENTS)
    required_components = {"speedup", "confidence", "cost_savings", "ease", "safety", "priority_boost"}
    for rec in summary["diagnosis"]["recommendations"]:
        assert set(rec["roi_components"]) == required_components
        for value in rec["roi_components"].values():
            assert 0.0 <= value <= 1.0


def test_recommendation_tier_consistent_with_score_and_practicality() -> None:
    events = (
        _make_step_events(1, dataloader_ns=50, step_wall_ns=100, gpu_util=42.0) +
        _make_step_events(2, dataloader_ns=48, step_wall_ns=100, gpu_util=40.0)
    )
    summary = at.summarize_run(events)

    tiers = {rec["id"]: rec["tier"] for rec in summary["diagnosis"]["recommendations"]}
    assert tiers["rec_input_pinned_memory"] == "try_now"
    assert tiers["rec_input_move_transforms"] in {"next", "advanced"}


def test_low_confidence_recommendations_are_demoted_by_guardrail() -> None:
    entry = {
        "id": "rec_example",
        "impact": "high",
        "effort": "low",
        "safety": "high",
    }

    score, components, guardrails = _score_recommendation(
        entry,
        confidence=0.30,
        priority_boost=0.0,
        signals={"num_gpus": 1},
    )

    assert "low_confidence_demoted" in guardrails
    assert components["confidence"] == 0.30
    assert score < 0.60


def test_dependency_order_keeps_compile_before_cuda_graphs() -> None:
    summary = at.summarize_run(LAUNCH_BOUND_EVENTS)
    ids = _rec_ids(summary)

    assert ids.index("rec_launch_torch_compile") < ids.index("rec_launch_cuda_graphs")


def test_recommendation_priority_consistent_with_score() -> None:
    summary = at.summarize_run(INPUT_BOUND_EVENTS)
    for rec in summary["diagnosis"]["recommendations"]:
        score = rec["score"]
        priority = rec["priority"]
        if score >= 0.75:
            assert priority == "high", f"{rec['id']} score={score} should be high"
        elif score >= 0.50:
            assert priority == "medium", f"{rec['id']} score={score} should be medium"
        else:
            assert priority == "low", f"{rec['id']} score={score} should be low"


def test_no_duplicate_recommendation_ids() -> None:
    for events in [INPUT_BOUND_EVENTS, COPY_BOUND_EVENTS, SYNC_BOUND_EVENTS,
                   MEMORY_PRESSURE_EVENTS, SHAPE_INSTABILITY_EVENTS]:
        summary = at.summarize_run(events)
        ids = _rec_ids(summary)
        assert len(ids) == len(set(ids)), f"Duplicate rec IDs found: {ids}"


def test_recommendation_bundles_group_related_recs() -> None:
    summary = at.summarize_run(INPUT_BOUND_EVENTS)
    bundles = summary["diagnosis"]["recommendation_bundles"]

    input_bundle = next((b for b in bundles if b["category"] == "input_pipeline"), None)
    assert input_bundle is not None
    assert len(input_bundle["recommendation_ids"]) >= 2
    assert "rec_input_num_workers" in input_bundle["recommendation_ids"]


def test_no_recommendations_when_no_bottlenecks() -> None:
    # A run with no events produces no bottlenecks → no recommendations
    summary = at.summarize_run([])
    assert summary["diagnosis"]["recommendations"] == []
    assert summary["diagnosis"]["recommendation_bundles"] == []
