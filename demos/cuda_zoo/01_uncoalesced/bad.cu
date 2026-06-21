// Demonstrates uncoalesced (strided / column-major) global memory access.
//
// The input is a WIDTH x HEIGHT matrix stored row-major. Each thread copies
// exactly one element, but reads COLUMN-major: adjacent threads (tid, tid+1)
// read addresses WIDTH elements apart, so a warp's 32 reads land in 32
// separate 128-byte sectors — ~32x the DRAM transactions of a coalesced read.
//
// This copies all N elements — the SAME total work as good.cu — differing
// ONLY in the access pattern. (The earlier version launched N/STRIDE threads
// and copied 32x less data than good.cu, so the comparison wasn't apples to
// apples; frx bench's kernel-time measurement exposed that.)
//
// Expected Fournex findings: uncoalesced_access

#include <cuda_runtime.h>
#include "frx_bench_harness.cuh"

#define WIDTH  1024
#define HEIGHT 1024
#define N (WIDTH * HEIGHT)

__global__ void strided_copy(const float* __restrict__ src,
                             float* __restrict__ dst,
                             int width, int height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < width * height) {
        int row = tid / width;
        int col = tid % width;
        dst[tid] = src[col * width + row];   // column-major read: strided by 'width'
    }
}

int main() {
    float *d_src, *d_dst;
    cudaMalloc(&d_src, N * sizeof(float));
    cudaMalloc(&d_dst, N * sizeof(float));
    cudaMemset(d_src, 0, N * sizeof(float));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    frx_bench([&] { strided_copy<<<blocks, threads>>>(d_src, d_dst, WIDTH, HEIGHT); });

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
