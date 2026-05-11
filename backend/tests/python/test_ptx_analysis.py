import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as at


# ── PTX fixtures ──────────────────────────────────────────────────────────────

PTX_SIMPLE = """
.version 8.0
.target sm_80
.address_size 64

.visible .entry simple_kernel(
    .param .u64 param0,
    .param .u64 param1
)
{
    .reg .pred  %p<4>;
    .reg .f32   %f<64>;
    .reg .b32   %r<32>;
    .reg .b64   %rd<16>;

    ld.param.u64    %rd0, [param0];
    ld.global.f32   %f0, [%rd0];
    fma.rn.f32      %f1, %f0, %f0, %f0;
    st.global.f32   [%rd0], %f1;
    ret;
}
"""

PTX_SPILL = """
.visible .entry spill_kernel() {
    .reg .f32   %f<256>;
    .reg .b64   %SP;
    .reg .b64   %SPL;
    .local .align 4 .b8 __local_depot0[512];
    ld.local.f32    %f0, [%SP+0];
    ld.local.f32    %f1, [%SP+4];
    fma.rn.f32      %f2, %f0, %f1, %f0;
    st.local.f32    [%SP+8], %f2;
    ret;
}
"""

PTX_FP64_SFU = """
.visible .entry mixed_kernel() {
    .reg .f64  %fd<16>;
    .reg .f32  %f<32>;
    fma.rn.f64      %fd0, %fd1, %fd2, %fd3;
    mul.f64         %fd4, %fd0, %fd1;
    sin.approx.f32  %f0, %f1;
    cos.approx.f32  %f2, %f3;
    ret;
}
"""

PTX_BRANCH = """
.visible .entry branch_kernel() {
    .reg .pred  %p<2>;
    .reg .b32   %r<8>;
    mov.u32         %r0, 0;
$L__loop_start:
    setp.lt.s32     %p0, %r0, 100;
    @%p0 bra        $L__loop_start;
    ret;
}
"""

PTX_SHARED = """
.visible .entry shared_kernel() {
    .reg .f32  %f<32>;
    .reg .b32  %r<16>;
    .shared .align 4 .b8 smem[1024];
    ld.shared.f32   %f0, [%r0];
    fma.rn.f32      %f1, %f0, %f0, %f0;
    st.shared.f32   [%r1], %f1;
    ret;
}
"""

PTX_TENSOR = """
.visible .entry wmma_kernel() {
    .reg .b32  %r<64>;
    wmma.load.a.sync.aligned.row.m16n16k16.global.f16 {%r0,%r1,%r2,%r3}, [%r10], 16;
    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%r20,%r21,%r22,%r23}, {%r0}, {%r4}, {%r8};
    ret;
}
"""

PTX_MULTI_KERNEL = PTX_SIMPLE + "\n" + PTX_SPILL

PTX_HIGH_GLOBAL = """
.visible .entry heavy_global() {
    .reg .f32  %f<32>;
    .reg .b64  %rd<8>;
    ld.global.f32 %f0, [%rd0];
    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd2];
    ld.global.f32 %f3, [%rd3];
    ld.global.f32 %f4, [%rd4];
    fma.rn.f32 %f5, %f0, %f1, %f2;
    st.global.f32 [%rd5], %f5;
    ret;
}
"""


# ── Register declarations ─────────────────────────────────────────────────────

def test_parse_ptx_register_declarations() -> None:
    kernels = at.parse_ptx_text(PTX_SIMPLE)
    assert len(kernels) == 1
    k = kernels[0]
    # pred excluded, f32=64, b32=32, b64=16*2=32 → total = 128
    assert k.register_breakdown["f32"] == 64
    assert k.register_breakdown["b32"] == 32
    assert k.register_breakdown["b64"] == 16
    assert "pred" not in k.register_breakdown
    assert k.register_count == 64 + 32 + 16 * 2  # 128


def test_parse_ptx_register_64bit_counts_double() -> None:
    text = """
    .visible .entry wide_kernel() {
        .reg .b64 %rd<8>;
        .reg .f64 %fd<4>;
        ret;
    }
    """
    kernels = at.parse_ptx_text(text)
    k = kernels[0]
    # b64: 8*2=16, f64: 4*2=8 → total 24
    assert k.register_count == 24


def test_parse_ptx_pred_registers_excluded() -> None:
    text = """
    .visible .entry pred_kernel() {
        .reg .pred %p<16>;
        .reg .f32  %f<8>;
        ret;
    }
    """
    kernels = at.parse_ptx_text(text)
    k = kernels[0]
    assert "pred" not in k.register_breakdown
    assert k.register_count == 8  # only f32 counts


# ── Spill detection ───────────────────────────────────────────────────────────

def test_parse_ptx_spill_detection() -> None:
    kernels = at.parse_ptx_text(PTX_SPILL)
    k = kernels[0]
    assert k.has_register_spills is True
    assert k.local_memory_bytes == 512
    assert k.spill_load_count == 2
    assert k.spill_store_count == 1


def test_parse_ptx_no_spills_simple_kernel() -> None:
    kernels = at.parse_ptx_text(PTX_SIMPLE)
    k = kernels[0]
    assert k.has_register_spills is False
    assert k.local_memory_bytes == 0
    assert k.spill_load_count == 0
    assert k.spill_store_count == 0


# ── Instruction mix ───────────────────────────────────────────────────────────

def test_instruction_mix_counts_global_memory() -> None:
    kernels = at.parse_ptx_text(PTX_SIMPLE)
    k = kernels[0]
    assert k.global_load_count == 1
    assert k.global_store_count == 1
    assert k.instruction_mix["global_loads"] == 1
    assert k.instruction_mix["global_stores"] == 1


def test_instruction_mix_counts_shared_memory() -> None:
    kernels = at.parse_ptx_text(PTX_SHARED)
    k = kernels[0]
    assert k.shared_load_count == 1
    assert k.shared_store_count == 1


def test_instruction_mix_fp32_ops() -> None:
    kernels = at.parse_ptx_text(PTX_SIMPLE)
    k = kernels[0]
    assert k.instruction_mix.get("fp32_ops", 0) >= 1  # fma.rn.f32


def test_instruction_mix_fp64_ops() -> None:
    kernels = at.parse_ptx_text(PTX_FP64_SFU)
    k = kernels[0]
    assert k.instruction_mix.get("fp64_ops", 0) == 2  # fma.f64 + mul.f64


def test_instruction_mix_special_func_ops() -> None:
    kernels = at.parse_ptx_text(PTX_FP64_SFU)
    k = kernels[0]
    assert k.instruction_mix.get("special_func", 0) == 2  # sin + cos


# ── Capability flags ──────────────────────────────────────────────────────────

def test_fp64_detection() -> None:
    kernels = at.parse_ptx_text(PTX_FP64_SFU)
    assert kernels[0].has_fp64 is True


def test_no_fp64_simple_kernel() -> None:
    kernels = at.parse_ptx_text(PTX_SIMPLE)
    assert kernels[0].has_fp64 is False


def test_sfu_detection() -> None:
    kernels = at.parse_ptx_text(PTX_FP64_SFU)
    assert kernels[0].has_special_function_ops is True


def test_tensor_ops_detection() -> None:
    kernels = at.parse_ptx_text(PTX_TENSOR)
    assert kernels[0].has_tensor_ops is True


# ── Control flow ─────────────────────────────────────────────────────────────

def test_conditional_branch_count() -> None:
    kernels = at.parse_ptx_text(PTX_BRANCH)
    k = kernels[0]
    assert k.conditional_branch_count == 1
    assert k.branch_count >= 1


def test_estimated_loop_count_back_edge() -> None:
    kernels = at.parse_ptx_text(PTX_BRANCH)
    k = kernels[0]
    assert k.estimated_loop_count >= 1  # @%p0 bra $L__loop_start is a back-edge


# ── Findings ──────────────────────────────────────────────────────────────────

def test_finding_register_spills_detected() -> None:
    kernels = at.parse_ptx_text(PTX_SPILL)
    codes = {f["code"] for f in kernels[0].findings}
    assert "register_spills_detected" in codes


def test_finding_fp64_detected() -> None:
    kernels = at.parse_ptx_text(PTX_FP64_SFU)
    codes = {f["code"] for f in kernels[0].findings}
    assert "fp64_detected" in codes


def test_finding_very_high_register_count() -> None:
    text = """
    .visible .entry fat_kernel() {
        .reg .f32 %f<130>;
        ret;
    }
    """
    kernels = at.parse_ptx_text(text)
    codes = {f["code"] for f in kernels[0].findings}
    assert "very_high_register_count" in codes


def test_finding_no_spills_clean_kernel() -> None:
    kernels = at.parse_ptx_text(PTX_SIMPLE)
    codes = {f["code"] for f in kernels[0].findings}
    assert "register_spills_detected" not in codes


def test_finding_high_global_memory_ratio() -> None:
    kernels = at.parse_ptx_text(PTX_HIGH_GLOBAL)
    codes = {f["code"] for f in kernels[0].findings}
    assert "high_global_memory_ratio" in codes


def test_finding_tensor_ops_info() -> None:
    kernels = at.parse_ptx_text(PTX_TENSOR)
    codes = {f["code"] for f in kernels[0].findings}
    assert "tensor_ops_detected" in codes


def test_all_findings_have_required_fields() -> None:
    for ptx in [PTX_SIMPLE, PTX_SPILL, PTX_FP64_SFU, PTX_BRANCH]:
        kernels = at.parse_ptx_text(ptx)
        for k in kernels:
            for finding in k.findings:
                assert "severity" in finding
                assert "code" in finding
                assert "message" in finding
                assert "suggestion" in finding
                assert finding["severity"] in {"high", "medium", "low"}


# ── analyze_ptx_text (full pipeline) ─────────────────────────────────────────

def test_analyze_ptx_text_multi_kernel() -> None:
    result = at.analyze_ptx_text(PTX_MULTI_KERNEL)
    assert result["schema"] == "ptx_analysis_v1"
    assert result["kernel_count"] == 2


def test_analyze_ptx_text_extracts_version_and_target() -> None:
    result = at.analyze_ptx_text(PTX_SIMPLE)
    assert result["ptx_version"] == "8.0"
    assert result["target"] == "sm_80"


def test_analyze_ptx_text_run_summary_spills() -> None:
    result = at.analyze_ptx_text(PTX_SPILL)
    summary = result["run_summary"]
    assert summary["any_spills"] is True
    assert summary["total_spill_loads"] >= 1
    assert summary["total_spill_stores"] >= 1


def test_analyze_ptx_text_run_summary_no_spills() -> None:
    result = at.analyze_ptx_text(PTX_SIMPLE)
    summary = result["run_summary"]
    assert summary["any_spills"] is False
    assert summary["has_fp64"] is False


def test_analyze_ptx_text_empty_returns_zero_kernels() -> None:
    result = at.analyze_ptx_text("")
    assert result["kernel_count"] == 0
    assert result["findings"] == []
