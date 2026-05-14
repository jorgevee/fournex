// Double-precision polynomial kernel.
// The arithmetic operations (mul.f64, fma.rn.f64, add.f64) appear directly in PTX,
// triggering the fp64_detected finding.
//
// Expected PTX analysis:
//   finding : fp64_detected  (arithmetic .f64 ops present)
//   rec     : rec_ptx_reduce_fp64

__global__ void poly_fp64(const double* __restrict__ in,
                            double* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double x = in[i];
        out[i] = x * x * x + 2.0 * x * x + x + 1.0;
    }
}
