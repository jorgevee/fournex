// Single-precision polynomial kernel — the fp32 equivalent of fp64_compute.cu.
// All arithmetic uses .f32 ops; no .f64 instructions are emitted.
//
// Expected PTX analysis:
//   NO fp64_detected finding
//   NO rec_ptx_reduce_fp64 recommendation

__global__ void poly_fp32(const float* __restrict__ in,
                            float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = x * x * x + 2.f * x * x + x + 1.f;
    }
}
