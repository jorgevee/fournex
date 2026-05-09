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
