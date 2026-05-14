// Register-pressure kernel — compiled with -maxrregcount 8 in the test suite.
// 10+ named float variables (a..m) exceed the 8-register limit, forcing nvcc to
// spill values to local memory (.local .align .b8 declaration + st.local/ld.local).
//
// Expected PTX analysis (when compiled with -maxrregcount 8):
//   finding : register_spills_detected  (.local memory declaration present)
//   rec     : rec_ptx_reduce_register_pressure

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
