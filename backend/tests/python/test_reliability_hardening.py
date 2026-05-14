"""Regression tests for the CUDA reliability hardening changes.

Four gaps closed by the hardening, each with a dedicated test group:

  1. v2 vectorized global load weighting
     ld.global.v2.* counts as 2 scalar loads; ratio detection still fires
     even when the compiler emits wide load instructions.

  2. Local FP64 data movement
     st.local.f64 / ld.local.f64 trigger fp64_data_movement_detected
     independent of FP64 arithmetic presence.

  3. FP64 arithmetic + data movement coexistence
     Both fp64_detected and fp64_data_movement_detected appear when a kernel
     has arithmetic FP64 ops and FP64 memory ops in the same body.

  4. Vectorized FP64 global load — intersection of two new features
     ld.global.v2.f64 applies weight-2 to BOTH global_load_count and
     fp64_data_movement_count.

NCU metric alias additions:
  5. sm__issue_active.avg.pct_of_peak_sustained_elapsed → issue_slot_utilization_pct
  6. smsp__warp_issue_stalled_memory_throttle_per_warp_active.pct → warp_stall_breakdown["memory_throttle"]

diagnostic_scope fields:
  7. PTX confidence "medium" / "low" depending on whether kernels were parsed
  8. NCU confidence "high" when metric data is present
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as at


# ── PTX fixtures ──────────────────────────────────────────────────────────────

# 4 × ld.global.v2.f32 (weight-2 each) + add.f32 + st.global.f32 + ret = 7 raw lines
# instruction_count=7, global_load_count=8, global_store_count=1, ratio=9/7≈1.29
PTX_V2_GLOBAL = """
.visible .entry v2_global_kernel() {
    .reg .f32  %f<16>;
    .reg .b64  %rd<4>;
    ld.global.v2.f32 {%f0, %f1}, [%rd0];
    ld.global.v2.f32 {%f2, %f3}, [%rd0+8];
    ld.global.v2.f32 {%f4, %f5}, [%rd0+16];
    ld.global.v2.f32 {%f6, %f7}, [%rd0+24];
    add.f32 %f8, %f0, %f2;
    st.global.f32 [%rd1], %f8;
    ret;
}
"""

# 2 × st.local.f64 + 1 × ld.local.f64 + add.f32 + ret = 5 raw lines
# has_fp64=False, fp64_data_movement_count=3, triggers fp64_data_movement_detected only
PTX_FP64_LOCAL_MOVEMENT = """
.visible .entry fp64_local_rw() {
    .reg .f64  %fd<4>;
    .reg .f32  %f<4>;
    .reg .b64  %rd<4>;
    st.local.f64 [%rd0], %fd0;
    st.local.f64 [%rd0+8], %fd1;
    ld.local.f64 %fd2, [%rd0];
    add.f32 %f0, %f1, %f2;
    ret;
}
"""

# ld.global.f64 + mul.f64 (arithmetic) + st.global.f64 + ret = 4 raw lines
# has_fp64=True (mul.f64), has_fp64_data_movement=True (ld/st .f64), fp64_data_movement_count=2
PTX_FP64_BOTH = """
.visible .entry fp64_full() {
    .reg .f64  %fd<8>;
    .reg .b64  %rd<4>;
    ld.global.f64 %fd0, [%rd0];
    mul.f64 %fd1, %fd0, %fd0;
    st.global.f64 [%rd1], %fd1;
    ret;
}
"""

# 1 × ld.global.v2.f64 + add.f32 + ret = 3 raw lines
# instruction_count=3, global_load_count=2 (v2 weight), fp64_data_movement_count=2 (same weight)
PTX_V2_FP64_GLOBAL = """
.visible .entry v2_fp64_copy() {
    .reg .f64  %fd<4>;
    .reg .f32  %f<4>;
    .reg .b64  %rd<4>;
    ld.global.v2.f64 {%fd0, %fd1}, [%rd0];
    add.f32 %f0, %f1, %f2;
    ret;
}
"""


# ── Group 1: v2 vectorized global load weighting ──────────────────────────────

def test_v2_global_load_instruction_count_and_weighted_count() -> None:
    kernels = at.parse_ptx_text(PTX_V2_GLOBAL)
    k = kernels[0]
    assert k.instruction_count == 7, (
        f"Expected 7 raw instruction lines (4 v2 + add + st + ret), got {k.instruction_count}"
    )
    assert k.global_load_count == 8, (
        f"Expected 8 weighted global loads (4 × v2 weight-2), got {k.global_load_count}"
    )
    assert k.instruction_mix["global_loads"] == 8


def test_v2_global_triggers_high_global_memory_ratio_finding() -> None:
    kernels = at.parse_ptx_text(PTX_V2_GLOBAL)
    codes = {f["code"] for f in kernels[0].findings}
    assert "high_global_memory_ratio" in codes, (
        f"Expected high_global_memory_ratio with v2 loads (ratio≈1.29); got codes={codes}"
    )


# ── Group 2: Local FP64 data movement ────────────────────────────────────────

def test_local_fp64_ops_trigger_data_movement_finding_not_fp64_detected() -> None:
    kernels = at.parse_ptx_text(PTX_FP64_LOCAL_MOVEMENT)
    k = kernels[0]
    codes = {f["code"] for f in k.findings}

    assert k.has_fp64 is False, (
        "st.local.f64 / ld.local.f64 should not set has_fp64 — "
        "they match the local_stores/local_loads pattern before .f64"
    )
    assert k.has_fp64_data_movement is True
    assert k.fp64_data_movement_count == 3, (
        f"Expected 3 local FP64 ops (2 st.local + 1 ld.local), got {k.fp64_data_movement_count}"
    )
    assert "fp64_data_movement_detected" in codes, f"Expected fp64_data_movement_detected; got {codes}"
    assert "fp64_detected" not in codes


# ── Group 3: FP64 arithmetic + data movement coexistence ─────────────────────

def test_fp64_arithmetic_and_data_movement_both_produce_findings() -> None:
    kernels = at.parse_ptx_text(PTX_FP64_BOTH)
    k = kernels[0]
    codes = {f["code"] for f in k.findings}

    assert k.has_fp64 is True, "mul.f64 should set has_fp64=True"
    assert k.has_fp64_data_movement is True, "ld/st .global.f64 should set has_fp64_data_movement=True"
    assert k.fp64_data_movement_count == 2
    assert "fp64_detected" in codes, f"Expected fp64_detected; got {codes}"
    assert "fp64_data_movement_detected" in codes, f"Expected fp64_data_movement_detected; got {codes}"


# ── Group 4: Vectorized FP64 global load (both weights applied) ───────────────

def test_vectorized_fp64_global_load_counted_in_both_categories() -> None:
    kernels = at.parse_ptx_text(PTX_V2_FP64_GLOBAL)
    k = kernels[0]

    assert k.instruction_count == 3, (
        f"Expected 3 raw lines (v2.f64 + add.f32 + ret), got {k.instruction_count}"
    )
    assert k.global_load_count == 2, (
        f"Expected 2 weighted global loads (1 v2 instruction × weight-2), got {k.global_load_count}"
    )
    assert k.fp64_data_movement_count == 2, (
        f"Expected 2 fp64 data movement ops (same weight-2 applied because .f64), "
        f"got {k.fp64_data_movement_count}"
    )
    assert k.has_fp64 is False, "No FP64 arithmetic — ld.global matches before .f64"
    assert k.has_fp64_data_movement is True


# ── Group 5: NCU alias — sm__issue_active elapsed variant ────────────────────

def test_sm_issue_active_elapsed_variant_maps_to_issue_slot_utilization() -> None:
    text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker,sm__issue_active.avg.pct_of_peak_sustained_elapsed,%,48.5",
    ])
    summary = at.parse_nsight_compute_csv_text(text)[0]
    assert summary.issue_slot_utilization_pct == 48.5, (
        f"sm__issue_active...elapsed should alias to issue_slot_utilization_pct; "
        f"got {summary.issue_slot_utilization_pct}"
    )


# ── Group 6: NCU alias — per_warp_active memory_throttle stall ───────────────

def test_per_warp_active_memory_throttle_alias_maps_to_stall_breakdown() -> None:
    text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker,smsp__warp_issue_stalled_memory_throttle_per_warp_active.pct,%,22.0",
    ])
    summary = at.parse_nsight_compute_csv_text(text)[0]
    assert summary.warp_stall_breakdown.get("memory_throttle") == 22.0, (
        f"smsp__warp_issue_stalled_memory_throttle_per_warp_active.pct should map to "
        f"warp_stall_breakdown['memory_throttle']; got {summary.warp_stall_breakdown}"
    )
    assert summary.dominant_warp_stall == "memory_throttle"


# ── Group 7: diagnostic_scope fields ─────────────────────────────────────────

def test_ptx_diagnostic_scope_confidence_is_medium_when_kernels_parsed() -> None:
    result = at.analyze_ptx_text(PTX_V2_GLOBAL)
    scope = result["diagnostic_scope"]
    assert scope["type"] == "static_ptx"
    assert scope["confidence"] == "medium", (
        f"Expected confidence='medium' when kernels are present; got '{scope['confidence']}'"
    )


def test_ptx_diagnostic_scope_confidence_is_low_when_no_kernels() -> None:
    result = at.analyze_ptx_text("")
    scope = result["diagnostic_scope"]
    assert scope["type"] == "static_ptx"
    assert scope["confidence"] == "low", (
        f"Expected confidence='low' when no kernels parsed; got '{scope['confidence']}'"
    )


def test_ptx_diagnostic_scope_message_mentions_ncu_or_benchmark() -> None:
    result = at.analyze_ptx_text(PTX_V2_GLOBAL)
    msg = result["diagnostic_scope"]["message"]
    assert "Nsight Compute" in msg or "benchmark" in msg, (
        f"Expected message to mention 'Nsight Compute' or 'benchmark'; got: {msg!r}"
    )


def test_ncu_diagnostic_scope_high_confidence_when_data_present() -> None:
    text = "\n".join([
        "Kernel Name,Metric Name,Metric Unit,Metric Value",
        "ker,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,80.0",
    ])
    result = at.analyze_ncu_csv_text(text)
    scope = result["diagnostic_scope"]
    assert scope["type"] == "measured_ncu"
    assert scope["confidence"] == "high", (
        f"Expected confidence='high' when NCU data is present; got '{scope['confidence']}'"
    )
