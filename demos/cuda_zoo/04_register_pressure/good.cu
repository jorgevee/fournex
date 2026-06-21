// Low register pressure: the same computation with one live accumulator chain.
//
// Instead of holding FEAT accumulators live at once (bad.cu, which spills to
// local memory at FEAT=256), this processes one feature chain to completion and
// folds it into a running sum. Only a couple of registers are live, so nothing
// spills and the kernel runs entirely out of registers.
//
// The result is identical to bad.cu, computed in a single kernel (no extra
// global round-trip). `nvcc -Xptxas -v` reports 0 spill bytes here vs hundreds
// in bad.cu — that spill traffic is the whole runtime difference.

#include <cuda_runtime.h>
#include "frx_bench_harness.cuh"

#define N     (1 << 20)
#define ITERS 32

__global__ void low_pressure(const float* __restrict__ in,
                             float* __restrict__ out,
                             int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float x = in[tid];
    float s = 0.0f;
    for (int j = 0; j < 256; j++) {        // one chain at a time -> ~2 live floats
        float a = x + 0.01f * j;
        for (int i = 0; i < ITERS; i++)
            a = a * 1.0001f + 0.0001f * j;
        s += a;
    }
    out[tid] = s;
}

int main() {
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemset(d_in, 0, N * sizeof(float));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    frx_bench([&] { low_pressure<<<blocks, threads>>>(d_in, d_out, N); });

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
