/*
 * good_linear_layer.cu
 *
 * Fixed version of bad_linear_layer.cu — same operation (C = ReLU(A @ B + bias))
 * with all four performance flaws corrected.
 *
 *   Fix 1 – Coalesced B access
 *     B is stored as B[K x N] (in_feature × out_feature).
 *     Access B[(t * TILE + threadIdx.y) * N + col]: adjacent threads vary col →
 *     stride-1 access → one cache line per warp.
 *
 *   Fix 2 – Shared-memory tiling
 *     Both A and B are loaded cooperatively into TILE×TILE shared-memory tiles.
 *     Each global element is loaded once and reused TILE times within a tile.
 *
 *   Fix 3 – (structural note)
 *     Full FP16 tensor-core tiling requires wmma or mma.sync intrinsics and
 *     changes the thread-block shape. Marked here as a follow-on step; the
 *     tiled FP32 kernel is the prerequisite.
 *
 *   Fix 4 – Legitimate __syncthreads() only
 *     Two syncs per tile iteration: one after loading (prevent use-before-store),
 *     one after computing (prevent overwrite-before-use). No sync inside the
 *     inner GEMM loop.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE 16

__global__ void good_linear_forward(
    const float* __restrict__ A,    // [M x K]
    const float* __restrict__ B,    // [K x N]  coalesced layout
    const float* __restrict__ bias, // [N]
    float*       __restrict__ C,    // [M x N]
    int M, int N, int K)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // Load tile of A: each thread loads A[row][t*TILE + threadIdx.x]
        sA[threadIdx.y][threadIdx.x] =
            (row < M && t * TILE + threadIdx.x < K)
            ? A[row * K + t * TILE + threadIdx.x]
            : 0.0f;

        // Load tile of B: B[K x N], access B[(t*TILE + threadIdx.y)][col]
        // Adjacent threads vary col → coalesced (stride 1)
        sB[threadIdx.y][threadIdx.x] =
            (t * TILE + threadIdx.y < K && col < N)
            ? B[(t * TILE + threadIdx.y) * N + col]
            : 0.0f;

        __syncthreads();  // wait for tile loads to complete

        for (int k = 0; k < TILE; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();  // wait before next tile overwrites shared mem
    }

    if (row < M && col < N) {
        acc += bias[col];
        C[row * N + col] = acc > 0.0f ? acc : 0.0f;
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

static void fill_random(float* d_ptr, int n)
{
    float* h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i)
        h[i] = (float)(rand() % 1000 - 500) / 500.0f;
    cudaMemcpy(d_ptr, h, n * sizeof(float), cudaMemcpyHostToDevice);
    free(h);
}

static void check(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// ── Main ───────────────────────────────────────────────────────────────────

int main()
{
    const int M = 1024;
    const int N = 1024;
    const int K = 256;

    float *d_A, *d_B, *d_bias, *d_C;
    check(cudaMalloc(&d_A,    M * K * sizeof(float)), "A");
    check(cudaMalloc(&d_B,    K * N * sizeof(float)), "B");
    check(cudaMalloc(&d_bias, N     * sizeof(float)), "bias");
    check(cudaMalloc(&d_C,    M * N * sizeof(float)), "C");

    fill_random(d_A,    M * K);
    fill_random(d_B,    K * N);
    fill_random(d_bias, N);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    for (int iter = 0; iter < 5; ++iter) {
        good_linear_forward<<<grid, block>>>(d_A, d_B, d_bias, d_C, M, N, K);
    }
    check(cudaDeviceSynchronize(), "sync");

    printf("good_linear_layer: M=%d N=%d K=%d  done.\n", M, N, K);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_C);
    return 0;
}
