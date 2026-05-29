"""Tests for framework_abstraction_tax.py — the Framework Abstraction Tax scorer."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex.framework_abstraction_tax import compute_framework_abstraction_tax
from fournex.analysis import summarize_step_scope
from analysis_bottleneck_golden_cases import LAUNCH_BOUND_TINY_KERNEL_EVENTS


def _run_summary(
    *,
    gpu_util=34.0,
    small_kernel_fraction=0.62,
    kernel_count_per_step=180.0,
    median_kernel_us=7.0,
    shape_volatility=0.05,
    profiler_windows=5,
):
    return {
        "average_gpu_utilization_pct": gpu_util,
        "small_kernel_fraction": small_kernel_fraction,
        "kernel_count_per_step": kernel_count_per_step,
        "median_cuda_kernel_duration_us": median_kernel_us,
        "shape_volatility_ratio": shape_volatility,
        "profiler_windows_exported": profiler_windows,
    }


def _bottleneck(label, **evidence):
    return {"label": label, "score": 0.5, "evidence": evidence}


def _names(tax):
    return [c["name"] for c in tax["contributors"]]


# ── Gating ───────────────────────────────────────────────────────────────────

def test_returns_none_without_profiler_telemetry():
    # NCU-only path: no profiler windows → nothing to reason about.
    summary = _run_summary(profiler_windows=0)
    assert compute_framework_abstraction_tax(summary, []) is None


# ── High-tax case ────────────────────────────────────────────────────────────

def test_high_tax_low_util_fragmented_stable_shapes():
    tax = compute_framework_abstraction_tax(_run_summary(), [])
    assert tax is not None
    assert tax["severity"] == "high"
    assert tax["score"] >= 45
    names = _names(tax)
    # Launch fragmentation is the measured headline driver.
    assert "Kernel launch fragmentation" in names
    # Stable shapes + heavy launch stream → inferred graph-capture opportunity.
    graph = next(c for c in tax["contributors"] if c["name"].startswith("Missing graph"))
    assert graph["inferred"] is True
    # Contributors ranked by points descending.
    points = [c["points"] for c in tax["contributors"]]
    assert points == sorted(points, reverse=True)


def test_inferred_contributors_are_flagged_and_measured_are_not():
    tax = compute_framework_abstraction_tax(_run_summary(), [])
    by_name = {c["name"]: c for c in tax["contributors"]}
    assert by_name["Kernel launch fragmentation"]["inferred"] is False
    for name, c in by_name.items():
        if "opportunity" in name:
            assert c["inferred"] is True


# ── Data-pipeline idle is subtracted out ─────────────────────────────────────

def test_input_bound_idle_does_not_count_as_framework_tax():
    # GPU is idle, but the idle is explained by the dataloader → low tax.
    summary = _run_summary(gpu_util=40.0)
    bottlenecks = [_bottleneck("input_bound", avg_dataloader_fraction=0.5)]
    tax = compute_framework_abstraction_tax(summary, bottlenecks)
    assert tax is not None
    assert tax["severity"] == "low"
    assert tax["score"] < 20
    # No mechanisms attributed when there is little unexplained idle.
    assert tax["contributors"] == []


def test_well_utilized_gpu_scores_low():
    # High GPU activity (e.g. a compute/memory-bound kernel) → little idle → low tax.
    summary = _run_summary(gpu_util=92.0)
    tax = compute_framework_abstraction_tax(summary, [])
    assert tax["severity"] == "low"
    assert tax["score"] < 20


# ── Dynamic shapes ───────────────────────────────────────────────────────────

def test_dynamic_shapes_surface_dispatch_overhead_not_graph_capture():
    summary = _run_summary(shape_volatility=0.7)
    tax = compute_framework_abstraction_tax(summary, [])
    names = _names(tax)
    assert "Dynamic-shape dispatch overhead" in names
    # Graph capture requires stable shapes, so it must NOT be suggested here.
    assert not any(n.startswith("Missing graph") for n in names)
    # The dynamic-shape contributor is a measured signal, not inferred.
    dyn = next(c for c in tax["contributors"] if c["name"].startswith("Dynamic-shape"))
    assert dyn["inferred"] is False


# ── Output shape ─────────────────────────────────────────────────────────────

def test_result_has_expected_keys_and_version():
    tax = compute_framework_abstraction_tax(_run_summary(), [])
    assert set(tax) >= {"score", "severity", "contributors", "evidence", "version"}
    assert tax["version"] == "fat_v1"
    assert 0 <= tax["score"] <= 100


# ── Wiring: the score reaches the analysis result ────────────────────────────

def test_tax_is_wired_into_summarize_step_scope():
    # Regression guard: a refactor of the assembler must not silently drop the
    # framework_abstraction_tax key from the analysis result.
    summary = summarize_step_scope(LAUNCH_BOUND_TINY_KERNEL_EVENTS)
    tax = summary.get("framework_abstraction_tax")
    assert tax is not None
    assert tax["severity"] == "high"
    assert any(c["name"] == "Kernel launch fragmentation" for c in tax["contributors"])
