#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA Error at %s:%d -> %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(err));                   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::fprintf(stderr, "cuBLAS Error at %s:%d -> code %d\n",         \
                         __FILE__, __LINE__, static_cast<int>(status));        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

struct GpuTimer {
    cudaEvent_t start_ev{};
    cudaEvent_t stop_ev{};

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_ev));
        CUDA_CHECK(cudaEventCreate(&stop_ev));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_ev);
        cudaEventDestroy(stop_ev);
    }
    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_ev, stream));
    }
    float stop_ms(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_ev, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));
        return ms;
    }
};

inline float max_abs_diff(const std::vector<float>& ref, const std::vector<float>& out) {
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_err = std::max(max_err, std::fabs(ref[i] - out[i]));
    }
    return max_err;
}
