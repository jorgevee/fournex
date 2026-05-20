// Demonstrates uncoalesced (strided) global memory access.
//
// Each thread reads from A[tid * STRIDE] — adjacent threads in a warp
// access addresses STRIDE*4 bytes apart. At STRIDE=32 one 128-byte
// transaction covers only a single word: 32× more DRAM transactions
// than a coalesced access pattern.
//
// Expected Fournex findings: uncoalesced_access

#include <cuda_runtime.h>
#include <stdint.h>

#define STRIDE 32
#define N (1 << 20)

__global__ void strided_copy(const float* __restrict__ src,
                              float* __restrict__ dst,
                              int n,
                              int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * stride < n)
        dst[tid] = src[tid * stride];   // strided_or_pitched: * stride inside []
}

int main() {
    float *d_src, *d_dst;
    cudaMalloc(&d_src, N * sizeof(float));
    cudaMalloc(&d_dst, (N / STRIDE) * sizeof(float));
    cudaMemset(d_src, 0, N * sizeof(float));

    int threads = 256;
    int blocks  = (N / STRIDE + threads - 1) / threads;
    strided_copy<<<blocks, threads>>>(d_src, d_dst, N, STRIDE);
    cudaDeviceSynchronize();

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
