// Shared-memory tiled FP32 matrix multiply.
//
// Each thread block loads TILE×TILE tiles of A and B into shared memory
// before accumulating. Each element of A is fetched from DRAM once per
// tile column, reducing DRAM traffic by TILE× vs. the naive version.
//
// TILE=16 → 256-thread blocks, 16× reuse factor, coalesced tile loads.
// Compare with bad.cu: same result, 16× fewer DRAM bytes.

#include <cuda_runtime.h>

#define M    1024
#define K    1024
#define N    1024
#define TILE 16

__global__ void tiled_gemm(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int m, int k, int n) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (k + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < m && a_col < k)
            ? A[row * k + a_col] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (b_row < k && col < n)
            ? B[b_row * n + col] : 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE; ++i)
            acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }

    if (row < m && col < n)
        C[row * n + col] = acc;
}

int main() {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    tiled_gemm<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
