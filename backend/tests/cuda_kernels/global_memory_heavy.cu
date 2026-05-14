// Intentionally memory-bound kernel: 64 global loads per thread with no shared memory.
// Stride-32 access pattern prevents vector-load optimisation by nvcc.
//
// Expected PTX analysis:
//   finding  : high_global_memory_ratio  (ratio ~47%, threshold > 0.40)
//   finding  : no_shared_memory_usage    (64 global loads, zero shared loads)
//   rec      : rec_ptx_stage_global_memory

__global__ void global_heavy(const float* __restrict__ in,
                               float* __restrict__ out) {
    int i = threadIdx.x;

    // 64 loads at stride-32 floats (128-byte step) from the same thread.
    // nvcc emits each as a separate ld.global.f32 with a constant byte offset,
    // so instruction_count ~ 136 and global_ops = 65 => ratio ~ 47 %.
    float s;
    s  = in[i +  0*32]; s += in[i +  1*32]; s += in[i +  2*32]; s += in[i +  3*32];
    s += in[i +  4*32]; s += in[i +  5*32]; s += in[i +  6*32]; s += in[i +  7*32];
    s += in[i +  8*32]; s += in[i +  9*32]; s += in[i + 10*32]; s += in[i + 11*32];
    s += in[i + 12*32]; s += in[i + 13*32]; s += in[i + 14*32]; s += in[i + 15*32];
    s += in[i + 16*32]; s += in[i + 17*32]; s += in[i + 18*32]; s += in[i + 19*32];
    s += in[i + 20*32]; s += in[i + 21*32]; s += in[i + 22*32]; s += in[i + 23*32];
    s += in[i + 24*32]; s += in[i + 25*32]; s += in[i + 26*32]; s += in[i + 27*32];
    s += in[i + 28*32]; s += in[i + 29*32]; s += in[i + 30*32]; s += in[i + 31*32];
    s += in[i + 32*32]; s += in[i + 33*32]; s += in[i + 34*32]; s += in[i + 35*32];
    s += in[i + 36*32]; s += in[i + 37*32]; s += in[i + 38*32]; s += in[i + 39*32];
    s += in[i + 40*32]; s += in[i + 41*32]; s += in[i + 42*32]; s += in[i + 43*32];
    s += in[i + 44*32]; s += in[i + 45*32]; s += in[i + 46*32]; s += in[i + 47*32];
    s += in[i + 48*32]; s += in[i + 49*32]; s += in[i + 50*32]; s += in[i + 51*32];
    s += in[i + 52*32]; s += in[i + 53*32]; s += in[i + 54*32]; s += in[i + 55*32];
    s += in[i + 56*32]; s += in[i + 57*32]; s += in[i + 58*32]; s += in[i + 59*32];
    s += in[i + 60*32]; s += in[i + 61*32]; s += in[i + 62*32]; s += in[i + 63*32];

    out[i] = s;
}

// Compiled only when building an executable for profiling (nvcc -DBUILD_EXECUTABLE ...).
// Not included in PTX-only compilation.
#ifdef BUILD_EXECUTABLE
int main(void) {
    const int N = 2048;  // 64 warps of 32 threads; fits on any modern device
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in,  N * sizeof(float));
    cudaMalloc((void**)&d_out, N * sizeof(float));
    global_heavy<<<N / 32, 32>>>(d_in, d_out);
    cudaDeviceSynchronize();
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
#endif
