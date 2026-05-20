// Low register pressure: same computation split across two kernel passes.
//
// Each kernel uses only 16 local scalars (v0..v15) and a partial sum.
// The compiler allocates fewer registers per thread, allowing more warps
// to reside on each SM → higher occupancy, better latency hiding.
//
// The global memory round-trip between the two passes adds a small
// bandwidth cost; the occupancy gain is typically the bigger win for
// compute-bound workloads.

#include <cuda_runtime.h>

#define N (1 << 20)

__global__ void low_pressure_first_half(const float* __restrict__ in,
                                         float* __restrict__ partial,
                                         int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float x = in[tid];

    float v0  = x * 1.00f + 0.00f;
    float v1  = x * 1.01f + 0.01f;
    float v2  = x * 1.02f + 0.02f;
    float v3  = x * 1.03f + 0.03f;
    float v4  = x * 1.04f + 0.04f;
    float v5  = x * 1.05f + 0.05f;
    float v6  = x * 1.06f + 0.06f;
    float v7  = x * 1.07f + 0.07f;
    float v8  = x * 1.08f + 0.08f;
    float v9  = x * 1.09f + 0.09f;
    float v10 = x * 1.10f + 0.10f;
    float v11 = x * 1.11f + 0.11f;
    float v12 = x * 1.12f + 0.12f;
    float v13 = x * 1.13f + 0.13f;
    float v14 = x * 1.14f + 0.14f;
    float v15 = x * 1.15f + 0.15f;

    partial[tid] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7
                 + v8 + v9 + v10 + v11 + v12 + v13 + v14 + v15;
}

__global__ void low_pressure_second_half(const float* __restrict__ in,
                                          const float* __restrict__ partial,
                                          float* __restrict__ out,
                                          int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float x = in[tid];

    float v16 = x * 1.16f + 0.16f;
    float v17 = x * 1.17f + 0.17f;
    float v18 = x * 1.18f + 0.18f;
    float v19 = x * 1.19f + 0.19f;
    float v20 = x * 1.20f + 0.20f;
    float v21 = x * 1.21f + 0.21f;
    float v22 = x * 1.22f + 0.22f;
    float v23 = x * 1.23f + 0.23f;
    float v24 = x * 1.24f + 0.24f;
    float v25 = x * 1.25f + 0.25f;
    float v26 = x * 1.26f + 0.26f;
    float v27 = x * 1.27f + 0.27f;
    float v28 = x * 1.28f + 0.28f;
    float v29 = x * 1.29f + 0.29f;
    float v30 = x * 1.30f + 0.30f;
    float v31 = x * 1.31f + 0.31f;

    out[tid] = partial[tid]
             + v16 + v17 + v18 + v19 + v20 + v21 + v22 + v23
             + v24 + v25 + v26 + v27 + v28 + v29 + v30 + v31;
}

int main() {
    float *d_in, *d_partial, *d_out;
    cudaMalloc(&d_in,      N * sizeof(float));
    cudaMalloc(&d_partial, N * sizeof(float));
    cudaMalloc(&d_out,     N * sizeof(float));
    cudaMemset(d_in, 0, N * sizeof(float));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    low_pressure_first_half <<<blocks, threads>>>(d_in, d_partial, N);
    low_pressure_second_half<<<blocks, threads>>>(d_in, d_partial, d_out, N);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_partial);
    cudaFree(d_out);
    return 0;
}
