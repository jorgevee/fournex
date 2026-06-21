// Naive FP32 matrix multiply — no shared memory, no tensor cores.
//
// Each thread computes one C[row][col] = sum_k A[row][k] * B[k][col].
// Every dot-product step reads A and B directly from global memory.
// For a 1024×1024 matrix each element of A is loaded 1024 times
// across all threads accessing the same column — DRAM traffic is O(N³).
//
// Expected Fournex findings: fp32_only_matmul, no_shared_memory_tiling

#include <cuda_runtime.h>
#include "frx_bench_harness.cuh"

#define M 1024
#define K 1024
#define N 1024

__global__ void naive_gemm(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float acc = 0.0f;
        for (int i = 0; i < k; ++i)
            acc += A[row * k + i] * B[i * n + col];   // raw global loads
        C[row * n + col] = acc;
    }
}

int main() {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    frx_bench([&] { naive_gemm<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N); });

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
