// Same reduction kernel as register_spills.cu — compiled without -maxrregcount.
// The compiler's natural register allocation is sufficient; no spills expected.
//
// Expected PTX analysis:
//   NO register_spills_detected finding
//   NO rec_ptx_reduce_register_pressure recommendation (unless natural reg count > 64)

__global__ void compute_reduce(const float* __restrict__ in,
                                 float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float a = in[i];
    float b = in[i ^ 1];
    float c = in[i ^ 2];
    float d = in[i ^ 3];

    float e = a + b;
    float f = c + d;
    float g = e * f;
    float h = sqrtf(g);

    float k = a * c + b * d;
    float m = e * h + f * g;

    out[i] = k + m + g + h + e + f;
}
