// frx_bench_harness.cuh — profiler-free kernel timing for `frx bench`.
//
// Why this exists: `frx bench` otherwise times the whole process (wall clock),
// which for a micro-kernel is dominated by ~160 ms of CUDA context init — so a
// 3x-faster kernel can still read ~1.0x. This header times only the kernel
// region with cudaEvents (GPU-side, warmup + repeated launches) and prints a
// sentinel line that `frx bench` parses:
//
//     FRX_KERNEL_US: <median microseconds per launch>
//
// Unlike NCU's duration, this number is NOT profiler-inflated: there is no
// serialization or replay, so the absolute value is production-representative.
//
// Usage — wrap your launch (or launches) in a lambda:
//
//     #include "frx_bench_harness.cuh"
//     ...
//     frx_bench([&] { my_kernel<<<blocks, threads>>>(args...); });
//
// `frx bench` puts this header on the nvcc include path automatically, so a
// bare `#include "frx_bench_harness.cuh"` resolves with no -I needed. To get a
// standalone copy: `frx bench --emit-harness`.
//
// Tunables (define before include, or pass -D to nvcc):
//   FRX_BENCH_WARMUP  (default 5)   launches discarded before timing
//   FRX_BENCH_ITERS   (default 50)  timed launches; the median is reported
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>

#ifndef FRX_BENCH_WARMUP
#define FRX_BENCH_WARMUP 5
#endif
#ifndef FRX_BENCH_ITERS
#define FRX_BENCH_ITERS 50
#endif

// Time `launch` (a callable that issues one or more kernel launches) and print
// the median per-launch GPU time in microseconds as a sentinel line on stdout.
template <typename LaunchFn>
inline void frx_bench(LaunchFn launch,
                      int warmup = FRX_BENCH_WARMUP,
                      int iters  = FRX_BENCH_ITERS) {
    if (iters < 1) iters = 1;
    if (warmup < 0) warmup = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup; ++i) launch();
    cudaDeviceSynchronize();

    std::vector<float> samples_ms;
    samples_ms.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        launch();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        samples_ms.push_back(ms);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Don't emit a timing sentinel for a failed run — let frx fall back.
        std::fprintf(stderr, "frx_bench: CUDA error: %s\n", cudaGetErrorString(err));
        return;
    }

    std::sort(samples_ms.begin(), samples_ms.end());
    float median_ms = samples_ms[samples_ms.size() / 2];
    std::printf("FRX_KERNEL_US: %.3f\n", median_ms * 1000.0);
    std::fflush(stdout);
}

// Statement-style convenience: FRX_BENCH(my_kernel<<<g, b>>>(args));
#define FRX_BENCH(...) frx_bench([&] { __VA_ARGS__; })
