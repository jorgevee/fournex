// Coalesced stride-1 global memory access.
//
// Adjacent threads in a warp access adjacent addresses. A single 128-byte
// DRAM transaction covers all 32 threads → peak DRAM efficiency.
//
// Copies the same N elements as bad.cu, differing ONLY in the access pattern:
// row-major (coalesced) here vs column-major (strided) there.

#include <cuda_runtime.h>
#include "frx_bench_harness.cuh"

#define WIDTH  1024
#define HEIGHT 1024
#define N (WIDTH * HEIGHT)

__global__ void coalesced_copy(const float* __restrict__ src,
                                float* __restrict__ dst,
                                int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        dst[tid] = src[tid];   // stride-1: fully coalesced
}

int main() {
    float *d_src, *d_dst;
    cudaMalloc(&d_src, N * sizeof(float));
    cudaMalloc(&d_dst, N * sizeof(float));
    cudaMemset(d_src, 0, N * sizeof(float));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    frx_bench([&] { coalesced_copy<<<blocks, threads>>>(d_src, d_dst, N); });

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
