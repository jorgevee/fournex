/*
 * bad_linear_layer.cu
 *
 * Simulates one forward pass of a fully-connected layer:
 *   C = ReLU(A @ B_T + bias)
 *   A      : [M x K]  (batch of M rows, K input features)
 *   B_T    : [N x K]  (N output features × K input features, stored transposed)
 *   bias   : [N]
 *   C      : [M x N]  (output)
 *
 * This is a realistic first-attempt implementation with four deliberate
 * performance flaws — the kind of code a developer writes before profiling:
 *
 *   Flaw 1 – Uncoalesced B access
 *     B_T is stored as B_T[out_feature][in_feature], i.e. row = output neuron.
 *     Each thread handles one (row, col) output element and loops over k:
 *       B_T[col * K + k]
 *     Adjacent threads in a warp vary over `col`, so their B_T addresses are
 *     K floats (= K*4 bytes) apart → stride-K access → high sectors/request.
 *
 *   Flaw 2 – No shared-memory tiling
 *     Every thread independently loads K floats from A and K floats from B_T.
 *     The entire K-dimension of A (per row) and B_T (per column) is reloaded
 *     from global memory for every output element → working set >> L1/L2 →
 *     low cache hit rates and high DRAM traffic.
 *
 *   Flaw 3 – FP32 only
 *     Weights and activations are float32. Tensor cores require FP16/BF16/TF32
 *     inputs; with plain FP32 GEMM they are idle.
 *
 *   Flaw 4 – Spurious __syncthreads() inside the reduction loop
 *     The developer added __syncthreads() "for safety" after each k-step,
 *     not realising there is no shared memory that needs synchronisation.
 *     This forces 256 barrier stalls per output element → warp stall / sync.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ── Kernel ─────────────────────────────────────────────────────────────────

__global__ void bad_linear_forward(
    const float* __restrict__ A,    // [M x K]
    const float* __restrict__ B_T,  // [N x K]  transposed weights
    const float* __restrict__ bias, // [N]
    float*       __restrict__ C,    // [M x N]
    int M, int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output feature index
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch index

    float acc = 0.0f;

    for (int k = 0; k < K; ++k) {
        // A[row][k] : row varies across blockDim.y → good locality within a column
        // B_T[col][k] : col varies across blockDim.x → stride-K access (Flaw 1)
        if (row < M && col < N) {
            acc += A[row * K + k] * B_T[col * K + k];
        }

        // Spurious barrier: no shared memory is in use (Flaw 4)
        __syncthreads();
    }

    if (row < M && col < N) {
        acc += bias[col];
        C[row * N + col] = acc > 0.0f ? acc : 0.0f;  // ReLU
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
    // Sizes typical of a hidden FC layer in a small transformer / MLP
    const int M = 1024;  // batch size
    const int N = 1024;  // output features
    const int K = 256;   // input features

    float *d_A, *d_B_T, *d_bias, *d_C;
    check(cudaMalloc(&d_A,    M * K * sizeof(float)), "A");
    check(cudaMalloc(&d_B_T,  N * K * sizeof(float)), "B_T");
    check(cudaMalloc(&d_bias, N     * sizeof(float)), "bias");
    check(cudaMalloc(&d_C,    M * N * sizeof(float)), "C");

    fill_random(d_A,    M * K);
    fill_random(d_B_T,  N * K);
    fill_random(d_bias, N);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Run several iterations so NCU has enough kernel launches to profile
    for (int iter = 0; iter < 5; ++iter) {
        bad_linear_forward<<<grid, block>>>(d_A, d_B_T, d_bias, d_C, M, N, K);
    }
    check(cudaDeviceSynchronize(), "sync");

    printf("bad_linear_layer: M=%d N=%d K=%d  done.\n", M, N, K);

    cudaFree(d_A);
    cudaFree(d_B_T);
    cudaFree(d_bias);
    cudaFree(d_C);
    return 0;
}
