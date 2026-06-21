// Register pressure that actually hurts: enough live per-thread state to SPILL.
//
// Each thread keeps FEAT independent accumulators live across an ITERS-step
// dependent update loop, then reduces them. At FEAT=256 the kernel needs more
// than the 255-register hardware limit, so the compiler spills accumulators to
// "local" memory (which is off-chip DRAM). Every inner-loop step then pays
// spill load/store traffic — far slower than staying in registers.
//
// IMPORTANT (the lesson, learned the hard way with frx bench): a *moderate*
// number of independent live values (e.g. 32, ~39 registers) is NOT slow on
// modern GPUs — the extra registers feed instruction-level parallelism that
// hides latency even at low occupancy. Register pressure only costs runtime
// once it forces SPILLING. Compile with `nvcc -Xptxas -v` and look for
// "spill stores / spill loads" to confirm.
//
// good.cu computes the identical result with ONE live accumulator at a time
// (no spills). Expected Fournex findings: high_register_pressure

#include <cuda_runtime.h>
#include "frx_bench_harness.cuh"

#define N     (1 << 20)
#define ITERS 32

__global__ void high_pressure(const float* __restrict__ in,
                              float* __restrict__ out,
                              int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float x = in[tid];
    float acc[256];                        // 256 live floats per thread -> spills
    #pragma unroll
    for (int j = 0; j < 256; j++)
        acc[j] = x + 0.01f * j;

    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < 256; j++)
            acc[j] = acc[j] * 1.0001f + 0.0001f * j;   // all 256 must stay live
    }

    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < 256; j++)
        s += acc[j];
    out[tid] = s;
}

int main() {
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemset(d_in, 0, N * sizeof(float));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    frx_bench([&] { high_pressure<<<blocks, threads>>>(d_in, d_out, N); });

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
