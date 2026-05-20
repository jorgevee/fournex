// Coalesced stride-1 global memory access.
//
// Adjacent threads in a warp access adjacent addresses. A single 128-byte
// DRAM transaction covers all 32 threads → peak DRAM efficiency.
//
// Compare with bad.cu: same logical copy, 32× better DRAM utilization.

#include <cuda_runtime.h>

#define N (1 << 20)

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
    coalesced_copy<<<blocks, threads>>>(d_src, d_dst, N);
    cudaDeviceSynchronize();

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
