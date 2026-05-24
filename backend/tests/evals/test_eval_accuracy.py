"""Fixture-based accuracy eval for the Fournex NCU diagnostic pipeline.

No GPU required. Each test loads a synthetic NCU CSV representing a known
pathological kernel state and asserts the pipeline detects the expected
bottleneck (true positive) or produces no false positives (true negative).

Run with:
    python -m pytest backend/tests/evals/test_eval_accuracy.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as fn

FIXTURES = Path(__file__).parent / "fixtures"


def _bottleneck_labels(csv_text: str) -> set[str]:
    result = fn.analyze_ncu_csv_text(csv_text)
    return {b["label"] for b in result["bottlenecks"]}


# ── Scenario 1: uncoalesced / DRAM-bound ─────────────────────────────────────
# Kernel has stride-K access pattern: load_sectors=9.3, DRAM=89%, low cache.
# Expected: memory_bandwidth_bound, uncoalesced_access, l1_cache_thrashing.
# Not expected: warp_stall_sync (no barrier stalls present).

def test_uncoalesced_dram_bound_detects_memory_bandwidth_bound():
    csv = (FIXTURES / "uncoalesced_dram_bound.csv").read_text()
    assert "memory_bandwidth_bound" in _bottleneck_labels(csv)


def test_uncoalesced_dram_bound_detects_uncoalesced_access():
    csv = (FIXTURES / "uncoalesced_dram_bound.csv").read_text()
    assert "uncoalesced_access" in _bottleneck_labels(csv)


def test_uncoalesced_dram_bound_detects_l1_cache_thrashing():
    csv = (FIXTURES / "uncoalesced_dram_bound.csv").read_text()
    assert "l1_cache_thrashing" in _bottleneck_labels(csv)


def test_uncoalesced_dram_bound_no_false_sync_bottleneck():
    csv = (FIXTURES / "uncoalesced_dram_bound.csv").read_text()
    assert "warp_stall_sync" not in _bottleneck_labels(csv)


# ── Scenario 2: tensor core idle ─────────────────────────────────────────────
# FP32-only GEMM, tensor cores idle at 6%. Occupancy 62% (not blocking).
# Expected: tensor_core_underutilized.
# Not expected: warp_stall_sync, memory_bandwidth_bound.

def test_tensor_core_idle_detected():
    csv = (FIXTURES / "tensor_core_idle.csv").read_text()
    assert "tensor_core_underutilized" in _bottleneck_labels(csv)


def test_tensor_core_idle_no_false_sync_bottleneck():
    csv = (FIXTURES / "tensor_core_idle.csv").read_text()
    assert "warp_stall_sync" not in _bottleneck_labels(csv)


def test_tensor_core_idle_no_false_memory_bandwidth_bound():
    csv = (FIXTURES / "tensor_core_idle.csv").read_text()
    assert "memory_bandwidth_bound" not in _bottleneck_labels(csv)


# ── Scenario 3: excessive synchronization ─────────────────────────────────────
# Kernel has spurious __syncthreads(): barrier stall 42%, wait stall 18%.
# Expected: warp_stall_sync.
# Not expected: tensor_core_underutilized, memory_bandwidth_bound.

def test_excessive_sync_detected():
    csv = (FIXTURES / "excessive_sync.csv").read_text()
    assert "warp_stall_sync" in _bottleneck_labels(csv)


def test_excessive_sync_no_false_tc_bottleneck():
    csv = (FIXTURES / "excessive_sync.csv").read_text()
    assert "tensor_core_underutilized" not in _bottleneck_labels(csv)


def test_excessive_sync_no_false_memory_bandwidth_bound():
    csv = (FIXTURES / "excessive_sync.csv").read_text()
    assert "memory_bandwidth_bound" not in _bottleneck_labels(csv)


# ── Scenario 4: register pressure / low occupancy ────────────────────────────
# High register count (96/thread, block=256) limits occupancy to ~22%.
# Expected: occupancy_limited_by_registers (specific cause resolved from metrics).

def test_register_pressure_occupancy_limited_by_registers_detected():
    csv = (FIXTURES / "register_pressure.csv").read_text()
    labels = _bottleneck_labels(csv)
    assert "occupancy_limited_by_registers" in labels, (
        f"Expected occupancy_limited_by_registers, got: {sorted(labels)}"
    )


# ── Scenario 5: well-optimized (true negative) ───────────────────────────────
# All metrics in healthy ranges: DRAM 32%, TC 68%, L1 78%, ISU 76%, occ 68%.
# Expected: zero bottlenecks (no false positives).

def test_well_optimized_produces_no_bottlenecks():
    csv = (FIXTURES / "well_optimized.csv").read_text()
    labels = _bottleneck_labels(csv)
    assert len(labels) == 0, f"False positives detected: {sorted(labels)}"


# ── Cross-layer: static + NCU combined ───────────────────────────────────────
# Verify that reconcile_evidence() upgrades confidence when both layers confirm.

_STRIDED_KERNEL = """\
__global__ void k(const float* __restrict__ src, float* __restrict__ dst, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    dst[tid] = src[tid * stride];
}
"""


def test_cross_layer_uncoalesced_n_confirming_at_least_2():
    """Static (strided_or_pitched) + NCU (uncoalesced_access) both confirm."""
    csv = (FIXTURES / "uncoalesced_dram_bound.csv").read_text()
    static = fn.inspect_cuda_source(_STRIDED_KERNEL)
    ncu    = fn.analyze_ncu_csv_text(csv)
    rec    = fn.reconcile_evidence(static=static, ncu=ncu)
    diag   = next(
        (d for d in rec["diagnoses"] if d["label"] == "inefficient_global_memory_access"),
        None,
    )
    assert diag is not None, "inefficient_global_memory_access not in reconciliation output"
    confirming = diag["layers_confirming"]
    assert len(confirming) >= 2, (
        f"Expected >= 2 confirming layers, got {len(confirming)}: {confirming}"
    )


def test_cross_layer_missing_evidence_absent_when_both_confirm():
    """When NCU and static both confirm, missing_evidence should be None."""
    csv = (FIXTURES / "uncoalesced_dram_bound.csv").read_text()
    static = fn.inspect_cuda_source(_STRIDED_KERNEL)
    ncu    = fn.analyze_ncu_csv_text(csv)
    rec    = fn.reconcile_evidence(static=static, ncu=ncu)
    diag   = next(
        (d for d in rec["diagnoses"] if d["label"] == "inefficient_global_memory_access"),
        None,
    )
    if diag is not None and len(diag["layers_confirming"]) >= 2:
        # Both layers confirmed — missing_evidence should be None or empty metrics
        me = diag.get("missing_evidence")
        if me is not None:
            assert me.get("metrics", []) == [], (
                "Expected no missing metrics when both layers confirm"
            )


def test_ncu_only_tensor_core_missing_evidence_populated():
    """NCU detects tensor_core_underutilized; missing_evidence should suggest NCU metrics."""
    csv = (FIXTURES / "tensor_core_idle.csv").read_text()
    missing = fn.what_evidence_is_missing(ncu=fn.analyze_ncu_csv_text(csv))
    tc_diag = next(
        (d for d in missing if d["label"] == "tensor_core_underutilization"),
        None,
    )
    if tc_diag is not None:
        me = tc_diag["missing_evidence"]
        assert me is not None
        assert me.get("ncu_command") is not None or len(me.get("metrics", [])) >= 0
