// Tree reduction with 3 barriers per loop iteration.
//
// Each reduction step uses three __syncthreads() calls:
//   (1) pre-read barrier   — unnecessary when only sdata[tid+stride] is read
//   (2) post-write barrier — required: ensure sdata[tid] is visible to all warps
//   (3) "drain" barrier    — unnecessary: no shared write follows it
//
// All 32 warps stall at every barrier even as fewer participate per step.
// Barriers 1 and 3 add pure latency with no correctness benefit.
//
// Expected Fournex findings: sync_inside_tight_loop
// (sync_in_loop_count >= 3: three __syncthreads() inside the for loop)

#include <cuda_runtime.h>

#define BLOCK 1024
#define N     (1 << 20)

__global__ void reduction_oversync(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK + tid;

    sdata[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();   // required: all threads finish loading

    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        __syncthreads();               // (1) redundant pre-read barrier
        float val = sdata[tid + stride];
        __syncthreads();               // (2) required: before write
        if (tid < stride)
            sdata[tid] += val;
        __syncthreads();               // (3) redundant drain barrier
    }

    if (tid == 0)
        out[blockIdx.x] = sdata[0];
}

int main() {
    float *d_in, *d_out;
    int blocks = (N + BLOCK - 1) / BLOCK;
    cudaMalloc(&d_in,  N      * sizeof(float));
    cudaMalloc(&d_out, blocks * sizeof(float));
    cudaMemset(d_in, 0, N * sizeof(float));

    reduction_oversync<<<blocks, BLOCK>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
