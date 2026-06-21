// Warp-shuffle reduction — no __syncthreads() inside the warp loop.
//
// Within a warp (32 threads) __shfl_down_sync() exchanges values with
// no barrier: all 32 lanes participate simultaneously in hardware.
// One __syncthreads() collects warp sums into shared memory; a second
// lets lane 0 of each warp sum the partial results.
//
// Total barriers: 2 (vs 10 in bad.cu) → far fewer stall cycles.

#include <cuda_runtime.h>
#include "frx_bench_harness.cuh"

#define BLOCK     1024
#define WARP_SIZE 32
#define N         (1 << 20)

__device__ float warp_reduce(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void reduction_shuffle(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int n) {
    __shared__ float warp_sums[BLOCK / WARP_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK + tid;

    float val = (gid < n) ? in[gid] : 0.0f;
    val = warp_reduce(val);          // no barrier inside warp

    if (tid % WARP_SIZE == 0)
        warp_sums[tid / WARP_SIZE] = val;
    __syncthreads();                 // barrier 1: collect warp sums

    if (tid < BLOCK / WARP_SIZE) {
        val = warp_sums[tid];
        val = warp_reduce(val);      // final reduce across warp sums
    }

    if (tid == 0)
        out[blockIdx.x] = val;
}

int main() {
    float *d_in, *d_out;
    int blocks = (N + BLOCK - 1) / BLOCK;
    cudaMalloc(&d_in,  N      * sizeof(float));
    cudaMalloc(&d_out, blocks * sizeof(float));
    cudaMemset(d_in, 0, N * sizeof(float));

    frx_bench([&] { reduction_shuffle<<<blocks, BLOCK>>>(d_in, d_out, N); });

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
