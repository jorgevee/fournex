// Intentionally flawed linear-layer forward pass: C = A @ B  (M x K) @ (K x N).
//
// Structural flaws — detected by Fournex static analysis:
//
//   1. No shared memory.  Every element of A and B is fetched directly from
//      global memory on every iteration of the K loop.  Each thread re-reads
//      the same A row element that its neighbor already fetched (no reuse).
//
//   2. Strided B access.  Thread (row, col) reads B[k * width + col].
//      Consecutive threads in the same warp share the same k but span
//      consecutive cols — so B reads ARE coalesced per iteration.  However
//      the per-iteration stride of width floats (one full row of B) means the
//      cache footprint per warp per tile is enormous, and without tiling the
//      same B elements are loaded once per A-row rather than once per tile.
//
//   3. Spurious __syncthreads().  There is no shared memory in this kernel,
//      so the barrier serves no purpose.  It only stalls the warp.
//
//   4. No bounds guard.  Out-of-bounds threads write to C unconditionally.
//
// Expected Fournex findings:
//   unnecessary_syncthreads     [medium]
//   missing_obvious_bounds_guard [medium]
//
// Expected memory_access_styles:
//   strided_or_pitched

__global__ void linear_layer_bad(
    const float* __restrict__ A,  // M x K, row-major
    const float* __restrict__ B,  // K x N, row-major
    float*       __restrict__ C,  // M x N, row-major
    int M, int K, int N,
    int width                     // == N; named 'width' so strided access is visible
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        // A[row * K + k]: coalesced across the K loop for a single thread,
        //                 but warp-level access to A rows is strided.
        // B[k * width + col]: one full B-row stride per K iteration.
        sum += A[row * K + k] * B[k * width + col];
    }

    // Spurious barrier: no shared memory exists; this stall is wasted cycles.
    __syncthreads();

    // No bounds check — threads outside the M x N tile write garbage.
    C[row * N + col] = sum;
}

#ifdef BUILD_EXECUTABLE
#include <cstdio>
int main(void) {
    const int M = 512, K = 256, N = 512;
    float *dA, *dB, *dC;
    cudaMalloc(&dA, M * K * sizeof(float));
    cudaMalloc(&dB, K * N * sizeof(float));
    cudaMalloc(&dC, M * N * sizeof(float));
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    linear_layer_bad<<<grid, block>>>(dA, dB, dC, M, K, N, N);
    cudaDeviceSynchronize();
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
#endif
