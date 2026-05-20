#include <cuda_runtime.h>
#include <stdio.h>

#define N (1 << 24)   // 16M floats
#define STRIDE 32     // adjacent threads read 32 elements apart — worst case for cache lines

__global__ void strided_read(const float* __restrict__ in, float* __restrict__ out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int count = n / STRIDE;
    if (tid < count) {
        float sum = 0.0f;
        for (int i = 0; i < STRIDE; i++)
            sum += in[tid + i * count];   // column-major stride — each thread's accesses are far apart
        out[tid] = sum;
    }
}

int main() {
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, (N / STRIDE) * sizeof(float));
    cudaMemset(d_in, 0, N * sizeof(float));

    int block = 256;
    int grid  = (N / STRIDE + block - 1) / block;

    strided_read<<<grid, block>>>(d_in, d_out, N);  // warmup
    cudaDeviceSynchronize();

    for (int i = 0; i < 5; i++)
        strided_read<<<grid, block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);
    printf("done\n");
}
