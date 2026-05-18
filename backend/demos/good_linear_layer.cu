// Optimised linear-layer forward pass: C = A @ B  (M x K) @ (K x N).
//
// Fixes applied relative to bad_linear_layer.cu:
//
//   1. Shared-memory tiling.  Each thread block loads a 16x16 tile of A and a
//      16x16 tile of B into on-chip shared memory before computing.  Each
//      global element is loaded once per tile iteration rather than once per
//      output element — reducing global traffic by a factor of TILE_DIM.
//
//   2. Coalesced global loads.  A tile is loaded so that thread (ty, tx) reads
//      A[row * K + tile*TILE + tx] — consecutive threads in x read consecutive
//      columns, which maps to consecutive global addresses.  B is loaded
//      similarly: thread (ty, tx) reads B[(tile*TILE + ty) * N + col].
//
//   3. Barrier only where needed.  __syncthreads() appears once after loading
//      the tiles (so all threads see consistent shared memory before compute)
//      and once after compute (before the next tile overwrites the buffers).
//      No spurious barriers.
//
//   4. Explicit bounds guard.  Threads outside the M x N output tile are
//      skipped; loads from A and B use ternary guards for the partial last tile.
//
//   5. Padding on tileB avoids bank conflicts.  The inner dimension is
//      TILE_DIM + 1 (17), which is not a multiple of 32, so consecutive warp
//      threads hit different banks when reading along a shared-memory column.
//
// Expected Fournex findings: none
//
// Expected memory_access_styles: shared_memory_tiling

#define TILE_DIM 16

__global__ void linear_layer_good(
    const float* __restrict__ A,  // M x K, row-major
    const float* __restrict__ B,  // K x N, row-major
    float*       __restrict__ C,  // M x N, row-major
    int M, int K, int N
) {
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM + 1];  // +1 padding avoids bank conflicts

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int ty  = threadIdx.y;
    int tx  = threadIdx.x;

    float sum = 0.0f;

    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_DIM + tx;
        int b_row = t * TILE_DIM + ty;

        // Coalesced load: consecutive tx → consecutive global column addresses.
        tileA[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        // Coalesced load: consecutive tx → consecutive global column addresses.
        tileB[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();  // wait until all threads have loaded their tile element

        for (int k = 0; k < TILE_DIM; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();  // prevent next tile load from overwriting in-use buffers
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

#ifdef BUILD_EXECUTABLE
#include <cstdio>
int main(void) {
    const int M = 512, K = 256, N = 512;
    float *dA, *dB, *dC;
    cudaMalloc(&dA, M * K * sizeof(float));
    cudaMalloc(&dB, K * N * sizeof(float));
    cudaMalloc(&dC, M * N * sizeof(float));
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    linear_layer_good<<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
#endif
