#pragma once
#include <cuda_runtime.h>

// Fused RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * weight
void launch_rmsnorm_naive(const float* d_x, const float* d_w, float* d_y, int rows, int cols, float eps, cudaStream_t stream = 0);
void launch_rmsnorm_warp_vec4(const float* d_x, const float* d_w, float* d_y, int rows, int cols, float eps, cudaStream_t stream = 0);

// Row-wise Softmax
void launch_softmax_naive(const float* d_x, float* d_y, int rows, int cols, cudaStream_t stream = 0);
void launch_softmax_online_warp(const float* d_x, float* d_y, int rows, int cols, cudaStream_t stream = 0);

// SGEMM: C = A * B (Row-major M x K * K x N -> M x N)
void launch_sgemm_naive(const float* d_A, const float* d_B, float* d_C, int M, int N, int K, cudaStream_t stream = 0);
void launch_sgemm_smem_tiled(const float* d_A, const float* d_B, float* d_C, int M, int N, int K, cudaStream_t stream = 0);
