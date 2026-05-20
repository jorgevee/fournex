import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as at


CUDA_SAMPLE = r"""
__global__ void tiled_sum(const float* x, float* y, int n) {
  __shared__ float tile[32][32];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    tile[threadIdx.y][threadIdx.x] = x[idx];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float acc = 0.0f;
    for (int stride = 16; stride > 0; stride >>= 1) {
      acc += tile[threadIdx.y][stride];
    }
    atomicAdd(&y[blockIdx.x], acc);
  }
}

void launch(const float* x, float* y, int n) {
  dim3 block(32, 8, 1);
  dim3 grid((n + 255) / 256);
  tiled_sum<<<grid, block, 0, stream>>>(x, y, n);
}
"""


def test_cuda_static_parser_detects_kernel_launch_and_patterns() -> None:
    report = at.inspect_cuda_source(CUDA_SAMPLE, filename="kernels.cu", gpu_model="A100")

    assert report["kernel_count"] == 1
    assert report["launch_count"] == 1
    kernel = report["kernels"][0]
    launch = report["launches"][0]

    assert kernel["name"] == "tiled_sum"
    assert "1d_grid_stride_index" in kernel["indexing_patterns"]
    assert "shared_memory_tiling" in kernel["memory_access_styles"]
    assert "atomicAdd(" in kernel["atomics"]
    assert "tree_reduction_loop" in kernel["reductions"]
    assert launch["kernel_name"] == "tiled_sum"
    assert launch["block_expr"] == "block"
    assert launch["stream_expr"] == "stream"


def test_shared_memory_heuristics_flag_bank_conflict_risk() -> None:
    report = at.inspect_cuda_source(CUDA_SAMPLE, filename="kernels.cu")

    shared = report["kernels"][0]["shared_memory"][0]
    finding_codes = {finding["code"] for finding in report["findings"]}

    assert shared["bytes"] == 4096
    assert shared["bank_conflict_risk"]
    assert "possible_shared_memory_bank_conflict" in finding_codes


def test_static_analyzer_flags_conditional_syncthreads() -> None:
    report = at.inspect_cuda_source(
        r"""
__global__ void bad_barrier(float* y) {
  int tid = threadIdx.x;
  if (tid < 16) {
    __syncthreads();
    y[tid] = 1.0f;
  }
}
""",
        filename="bad.cu",
    )

    codes = {finding["code"] for finding in report["findings"]}
    assert "conditional_syncthreads" in codes


def test_gpu_model_limits_feed_occupancy_and_launch_advisor() -> None:
    limits = at.device_limits_for_gpu("NVIDIA H100")
    estimate = at.estimate_occupancy(
        registers_per_thread=64,
        shared_memory_per_block_bytes=32768,
        threads_per_block=256,
        device_limits=limits,
    )
    report = at.inspect_cuda_source(CUDA_SAMPLE, filename="kernels.cu", gpu_model="NVIDIA H100")

    assert limits["shared_memory_per_sm_bytes"] > 200000
    assert estimate["occupancy_pct"] == 50.0
    advice = report["launch_advisor"][0]
    assert "Safe recommended starting configurations" in advice["notes"][0]
    assert {item["block_size"] for item in advice["candidate_block_sizes"]} == {128, 256, 512}


def test_parse_cuda_files_reads_cu_and_cuh(tmp_path: Path) -> None:
    path = tmp_path / "kernels.cuh"
    path.write_text(CUDA_SAMPLE, encoding="utf-8")

    report = at.inspect_cuda_files([path])

    assert report["kernel_count"] == 1
    assert report["kernels"][0]["filename"].endswith("kernels.cuh")


# ── New antipattern rules ──────────────────────────────────────────────────────

def _codes(source: str) -> set[str]:
    return {f["code"] for f in at.inspect_cuda_source(source)["findings"]}


# Memory: uncoalesced_access
def test_uncoalesced_access_detected_for_strided_pattern() -> None:
    src = """
__global__ void bad(float* A, float* B, int stride) {
    int i = threadIdx.x;
    B[i] = A[i * stride];
}
"""
    assert "uncoalesced_access" in _codes(src)


def test_uncoalesced_access_not_fired_for_coalesced() -> None:
    src = """
__global__ void good(float* A, float* B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    B[idx] = A[idx];
}
"""
    assert "uncoalesced_access" not in _codes(src)


# Memory: no_shared_memory_tiling
def test_no_shared_memory_tiling_detected_for_nested_loops() -> None:
    src = """
__global__ void naive_gemm(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
"""
    codes = _codes(src)
    assert "no_shared_memory_tiling" in codes


def test_no_shared_memory_tiling_not_fired_when_shared_present() -> None:
    src = """
__global__ void tiled(float* A, float* B, float* C, int N) {
    __shared__ float sA[16][16];
    for (int t = 0; t < N; t++) {
        for (int k = 0; k < 16; k++) {
            sA[k][threadIdx.x] = A[t + k];
        }
        __syncthreads();
    }
}
"""
    assert "no_shared_memory_tiling" not in _codes(src)


# Memory: missing_vectorized_loads
def test_missing_vectorized_loads_detected_for_simple_float_kernel() -> None:
    src = """
__global__ void scale(float* out, float* in, float factor) {
    int idx = threadIdx.x;
    out[idx] = in[idx] * factor;
}
"""
    assert "missing_vectorized_loads" in _codes(src)


def test_missing_vectorized_loads_not_fired_when_float4_used() -> None:
    src = """
__global__ void scale4(float4* out, float4* in) {
    int idx = threadIdx.x;
    out[idx] = in[idx];
}
"""
    assert "missing_vectorized_loads" not in _codes(src)


# Synchronization: sync_inside_tight_loop
def test_sync_inside_tight_loop_detected_for_many_syncthreads() -> None:
    src = """
__global__ void over_sync(float* A, int N) {
    __shared__ float s[64];
    for (int i = 0; i < N; i++) {
        s[threadIdx.x] = A[i];
        __syncthreads();
        A[i] = s[threadIdx.x] + 1.0f;
        __syncthreads();
        s[threadIdx.x] = A[i] * 2.0f;
        __syncthreads();
    }
}
"""
    assert "sync_inside_tight_loop" in _codes(src)


def test_sync_inside_tight_loop_not_fired_for_two_syncs() -> None:
    src = """
__global__ void tiled(float* A, int N) {
    __shared__ float sA[16][16];
    __shared__ float sB[16][16];
    for (int t = 0; t < N; t++) {
        sA[threadIdx.y][threadIdx.x] = A[t];
        sB[threadIdx.y][threadIdx.x] = A[t + N];
        __syncthreads();
        A[t] += sA[0][0] + sB[0][0];
        __syncthreads();
    }
}
"""
    assert "sync_inside_tight_loop" not in _codes(src)


# Synchronization: warp_level_sync_misuse
def test_warp_level_sync_misuse_detected() -> None:
    src = """
__global__ void misuse(float* A) {
    __shared__ float s[32];
    s[threadIdx.x] = A[threadIdx.x];
    __syncwarp();
    A[threadIdx.x] = s[31 - threadIdx.x];
}
"""
    assert "warp_level_sync_misuse" in _codes(src)


# Control flow: warp_divergence
def test_warp_divergence_detected_for_modulo_branch() -> None:
    src = """
__global__ void diverge(float* A) {
    if (threadIdx.x % 2 == 0) {
        A[threadIdx.x] = 1.0f;
    } else {
        A[threadIdx.x] = 0.0f;
    }
}
"""
    assert "warp_divergence" in _codes(src)


def test_warp_divergence_detected_for_bitmask_branch() -> None:
    src = """
__global__ void bitmask(float* A) {
    if (threadIdx.x & 1) {
        A[threadIdx.x] = 1.0f;
    }
}
"""
    assert "warp_divergence" in _codes(src)


def test_warp_divergence_not_fired_for_non_threadidx_branch() -> None:
    src = """
__global__ void safe_branch(float* A, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] = 1.0f;
    }
}
"""
    assert "warp_divergence" not in _codes(src)


# Control flow: excessive_branching
def test_excessive_branching_detected_for_many_ifs() -> None:
    src = """
__global__ void branchy(float* A, int n) {
    int idx = threadIdx.x;
    if (A[0] > 0) A[idx] += 1;
    if (A[1] > 0) A[idx] += 2;
    if (A[2] > 0) A[idx] += 3;
    if (A[3] > 0) A[idx] += 4;
    if (A[4] > 0) A[idx] += 5;
    if (A[5] > 0) A[idx] += 6;
    if (A[6] > 0) A[idx] += 7;
}
"""
    assert "excessive_branching" in _codes(src)


# Occupancy: high_register_pressure
def test_high_register_pressure_detected_for_many_locals() -> None:
    src = """
__global__ void heavy(float* A) {
    float a0 = A[0]; float a1 = A[1]; float a2 = A[2]; float a3 = A[3];
    float a4 = A[4]; float a5 = A[5]; float a6 = A[6]; float a7 = A[7];
    float b0 = A[8]; float b1 = A[9]; float b2 = A[10]; float b3 = A[11];
    float b4 = A[12]; float b5 = A[13]; float b6 = A[14]; float b7 = A[15];
    float c0 = a0 + b0; float c1 = a1 + b1; float c2 = a2 + b2;
    float c3 = a3 + b3; float c4 = a4 + b4; float c5 = a5 + b5;
    A[threadIdx.x] = c0 + c1 + c2 + c3 + c4 + c5;
}
"""
    assert "high_register_pressure" in _codes(src)


# Tensor cores: fp32_only_matmul
def test_fp32_only_matmul_detected_for_naive_gemm() -> None:
    src = """
__global__ void naive_gemm(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
"""
    assert "fp32_only_matmul" in _codes(src)


def test_fp32_only_matmul_not_fired_when_wmma_present() -> None:
    src = """
#include <mma.h>
using namespace nvcuda;
__global__ void tc_gemm(half* A, half* B, float* C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::load_matrix_sync(a_frag, A, 16);
}
"""
    assert "fp32_only_matmul" not in _codes(src)


# Tensor cores: missing_wmma_mma_path
def test_missing_wmma_mma_path_detected_for_fp16_without_tc() -> None:
    src = """
__global__ void fp16_gemm(__half* A, __half* B, __half* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __half sum = __float2half(0.0f);
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
"""
    assert "missing_wmma_mma_path" in _codes(src)


# Tensor cores: dimensions_not_tensor_core_friendly
def test_dimensions_not_tc_friendly_detected_for_odd_tile() -> None:
    src = """
__global__ void odd_tile(float* A) {
    __shared__ float tile[20][20];
    tile[threadIdx.y][threadIdx.x] = A[threadIdx.x];
    __syncthreads();
    A[threadIdx.x] = tile[threadIdx.y][threadIdx.x];
}
"""
    assert "dimensions_not_tensor_core_friendly" in _codes(src)


def test_dimensions_tc_friendly_not_fired_for_16x16_tile() -> None:
    src = """
__global__ void tc_tile(float* A) {
    __shared__ float tile[16][16];
    tile[threadIdx.y][threadIdx.x] = A[threadIdx.x];
    __syncthreads();
    A[threadIdx.x] = tile[threadIdx.y][threadIdx.x];
}
"""
    assert "dimensions_not_tensor_core_friendly" not in _codes(src)


# Launch-level: poor_block_size_unaligned / poor_block_size_subwarp
def test_poor_block_size_detected_for_non_warp_multiple() -> None:
    src = """
__global__ void kernel(float* A) { A[threadIdx.x] = 1.0f; }
void launch(float* A) { kernel<<<1, 100>>>(A); }
"""
    assert "poor_block_size_unaligned" in _codes(src)


def test_poor_block_size_high_severity_for_sub_warp_size() -> None:
    src = """
__global__ void kernel(float* A) { A[threadIdx.x] = 1.0f; }
void launch(float* A) { kernel<<<1, 16>>>(A); }
"""
    findings = at.inspect_cuda_source(src)["findings"]
    poor = [f for f in findings if f["code"] == "poor_block_size_subwarp"]
    assert poor and poor[0]["severity"] == "high"


# ── Strided alias detection ────────────────────────────────────────────────────

def test_uncoalesced_detected_via_alias_variable() -> None:
    # int idx = tid * stride; src[idx]  — stride keyword inside assignment, idx in subscript
    src = """
__global__ void strided_via_alias(const float* src, float* dst, int stride, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * stride;
    if (idx < n)
        dst[tid] = src[idx];
}
"""
    assert "uncoalesced_access" in _codes(src), (
        "Alias idx=tid*stride should trigger uncoalesced_access"
    )


def test_uncoalesced_detected_via_offset_alias() -> None:
    # size_t offset = row * pitch + col  — pitch keyword, subscript uses offset
    src = """
__global__ void pitched_read(const float* A, float* B, int pitch, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t offset = row * pitch + col;
    B[row * n + col] = A[offset];
}
"""
    assert "uncoalesced_access" in _codes(src), (
        "Alias offset=row*pitch+col should trigger uncoalesced_access"
    )


def test_uncoalesced_not_fired_for_unrelated_alias() -> None:
    # Variable named 'idx' but assigned from non-strided expression
    src = """
__global__ void coalesced_alias(const float* src, float* dst, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid;           // no stride keyword in RHS
    if (idx < n)
        dst[idx] = src[idx];
}
"""
    assert "uncoalesced_access" not in _codes(src), (
        "idx=tid (no stride) should not trigger uncoalesced_access"
    )


# ── sync_in_loop accurate detection ───────────────────────────────────────────

def test_sync_in_loop_false_positive_prevented() -> None:
    # 3 syncs in setup (before any loop) + loop with no syncs inside — should NOT fire
    src = """
__global__ void three_setup_syncs(float* A, int N) {
    __shared__ float s[64];
    s[threadIdx.x] = 0.0f;
    __syncthreads();
    s[threadIdx.x] = A[threadIdx.x];
    __syncthreads();
    __syncthreads();    // extra "safety" init sync

    for (int i = 0; i < N; i++) {
        A[i] += s[threadIdx.x];   // no sync inside
    }
}
"""
    assert "sync_inside_tight_loop" not in _codes(src), (
        "3 setup syncs + loop-without-sync should not fire sync_inside_tight_loop"
    )


def test_sync_in_loop_fires_for_three_in_loop_body() -> None:
    # 3 __syncthreads() inside the for loop body — the paranoid pattern
    src = """
__global__ void oversync(float* s, int N) {
    for (int stride = N / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        float v = s[threadIdx.x + stride];
        __syncthreads();
        if (threadIdx.x < stride) s[threadIdx.x] += v;
        __syncthreads();
    }
}
"""
    assert "sync_inside_tight_loop" in _codes(src), (
        "3 syncs inside for loop body should fire sync_inside_tight_loop"
    )


def test_poor_block_size_not_fired_for_256_threads() -> None:
    src = """
__global__ void kernel(float* A) { A[threadIdx.x] = 1.0f; }
void launch(float* A) { kernel<<<1, 256>>>(A); }
"""
    assert "poor_block_size_subwarp" not in _codes(src)
    assert "poor_block_size_unaligned" not in _codes(src)
