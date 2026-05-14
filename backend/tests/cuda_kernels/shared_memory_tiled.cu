// Shared-memory optimised version of the data-reduction pattern in global_memory_heavy.cu.
// One global load stages data into shared memory; all reduction reads use ld.shared.
//
// Expected PTX analysis:
//   global ratio ~ 1%  (2 global ops out of ~250+ instructions)
//   NO high_global_memory_ratio finding
//   NO rec_ptx_stage_global_memory recommendation

__global__ void shared_tiled(const float* __restrict__ in,
                               float* __restrict__ out, int n) {
    __shared__ float smem[256];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 1 global load per thread → stage into shared memory.
    smem[threadIdx.x] = (i < n) ? in[i] : 0.f;
    __syncthreads();

    // 32 ld.shared per thread — no global memory involved.
    float s = 0.f;
    for (int j = 0; j < 32; j++) {
        s += smem[(threadIdx.x + j) % 256];
    }

    // 1 global store.
    if (i < n) out[i] = s;
}
