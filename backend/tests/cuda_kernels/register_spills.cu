// Local-memory spill kernel.
// A volatile float array with runtime indexing cannot be register-allocated by
// any nvcc version, so the compiler always emits:
//   .local .align 4 .b8 __local_depot0[64]   (16 floats × 4 bytes)
//   st.local / ld.local instructions
// This is architecture-independent — no -maxrregcount flag needed.
//
// Expected PTX analysis:
//   finding : register_spills_detected  (.local memory declaration present)
//   rec     : rec_ptx_reduce_register_pressure

__global__ void compute_reduce(const float* __restrict__ in,
                                 float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // volatile prevents nvcc from promoting the array to registers.
    // Runtime indexing with threadIdx ensures the compiler cannot fold
    // accesses at compile time.
    volatile float arr[16];
    for (int j = 0; j < 16; j++) {
        arr[j] = in[(i + j) % n];
    }

    float sum = 0.f;
    for (int j = 0; j < 16; j++) {
        sum += arr[j];
    }
    out[i] = sum;
}
