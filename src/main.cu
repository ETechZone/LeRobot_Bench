#include "cuda_utils.cuh"
#include "kernels.cuh"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cfloat>

// CPU Reference Implementations
void cpu_rmsnorm(const std::vector<float>& x, const std::vector<float>& w, std::vector<float>& y, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        double sum_sq = 0.0;
        for (int c = 0; c < cols; ++c) {
            double v = x[r * cols + c];
            sum_sq += v * v;
        }
        float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / cols) + eps);
        for (int c = 0; c < cols; ++c) {
            y[r * cols + c] = x[r * cols + c] * inv_rms * w[c];
        }
    }
}

void cpu_softmax(const std::vector<float>& x, std::vector<float>& y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float max_v = -FLT_MAX;
        for (int c = 0; c < cols; ++c) max_v = std::max(max_v, x[r * cols + c]);
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) {
            float e = std::exp(x[r * cols + c] - max_v);
            y[r * cols + c] = e;
            sum += e;
        }
        for (int c = 0; c < cols; ++c) y[r * cols + c] /= static_cast<float>(sum);
    }
}

void cpu_sgemm(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += static_cast<double>(A[m * K + k]) * B[k * N + n];
            C[m * N + n] = static_cast<float>(acc);
        }
    }
}

template <typename Func>
float benchmark_ms(Func&& fn, int warmup = 10, int iters = 50) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) fn();
    return timer.stop_ms() / static_cast<float>(iters);
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== CUDA LLM Kernel Benchmark Suite ===\n");
    std::printf("Device: %s (SM %d.%d, Global Mem: %.2f GB)\n\n",
                prop.name, prop.major, prop.minor, prop.totalGlobalMem / 1e9);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 1. Fused RMSNorm Benchmark (LLM Hidden Dim = 4096, Tokens = 4096)
    {
        const int rows = 4096, cols = 4096;
        const float eps = 1e-5f;
        size_t bytes = rows * cols * sizeof(float);
        std::vector<float> h_x(rows * cols), h_w(cols), h_ref(rows * cols), h_out(rows * cols);
        for (auto& v : h_x) v = dist(rng);
        for (auto& v : h_w) v = dist(rng);
        cpu_rmsnorm(h_x, h_w, h_ref, rows, cols, eps);

        float *d_x, *d_w, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, bytes));
        CUDA_CHECK(cudaMalloc(&d_w, cols * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, bytes));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

        float ms_naive = benchmark_ms([&]() { launch_rmsnorm_naive(d_x, d_w, d_y, rows, cols, eps); });
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, bytes, cudaMemcpyDeviceToHost));
        float err_naive = max_abs_diff(h_ref, h_out);

        float ms_opt = benchmark_ms([&]() { launch_rmsnorm_warp_vec4(d_x, d_w, d_y, rows, cols, eps); });
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, bytes, cudaMemcpyDeviceToHost));
        float err_opt = max_abs_diff(h_ref, h_out);

        double gb_moved = (2.0 * bytes + cols * sizeof(float)) / 1e9;
        std::printf("[RMSNorm 4096x4096]\n");
        std::printf("  Naive:          %7.3f ms | %6.1f GB/s | max_err=%.2e\n", ms_naive, gb_moved / (ms_naive * 1e-3), err_naive);
        std::printf("  Warp+Float4:    %7.3f ms | %6.1f GB/s | max_err=%.2e (%.2fx speedup)\n\n",
                    ms_opt, gb_moved / (ms_opt * 1e-3), err_opt, ms_naive / ms_opt);

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_w));
        CUDA_CHECK(cudaFree(d_y));
    }

    // 2. Online Softmax Benchmark (4096 x 4096)
    {
        const int rows = 4096, cols = 4096;
        size_t bytes = rows * cols * sizeof(float);
        std::vector<float> h_x(rows * cols), h_ref(rows * cols), h_out(rows * cols);
        for (auto& v : h_x) v = dist(rng);
        cpu_softmax(h_x, h_ref, rows, cols);

        float *d_x, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, bytes));
        CUDA_CHECK(cudaMalloc(&d_y, bytes));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

        float ms_naive = benchmark_ms([&]() { launch_softmax_naive(d_x, d_y, rows, cols); });
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, bytes, cudaMemcpyDeviceToHost));
        float err_naive = max_abs_diff(h_ref, h_out);

        float ms_opt = benchmark_ms([&]() { launch_softmax_online_warp(d_x, d_y, rows, cols); });
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_y, bytes, cudaMemcpyDeviceToHost));
        float err_opt = max_abs_diff(h_ref, h_out);

        double gb_moved = (2.0 * bytes) / 1e9;
        std::printf("[Softmax 4096x4096]\n");
        std::printf("  Naive (3-pass): %7.3f ms | %6.1f GB/s | max_err=%.2e\n", ms_naive, gb_moved / (ms_naive * 1e-3), err_naive);
        std::printf("  Online Warp:    %7.3f ms | %6.1f GB/s | max_err=%.2e (%.2fx speedup)\n\n",
                    ms_opt, gb_moved / (ms_opt * 1e-3), err_opt, ms_naive / ms_opt);

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
    }

    // 3. Tiled SGEMM Benchmark (1024 x 1024 x 1024)
    {
        const int M = 1024, N = 1024, K = 1024;
        std::vector<float> h_A(M * K), h_B(K * N), h_ref(M * N), h_out(M * N);
        for (auto& v : h_A) v = dist(rng);
        for (auto& v : h_B) v = dist(rng);
        cpu_sgemm(h_A, h_B, h_ref, M, N, K);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

        float ms_naive = benchmark_ms([&]() { launch_sgemm_naive(d_A, d_B, d_C, M, N, K); });
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        float err_naive = max_abs_diff(h_ref, h_out);

        float ms_tiled = benchmark_ms([&]() { launch_sgemm_smem_tiled(d_A, d_B, d_C, M, N, K); });
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        float err_tiled = max_abs_diff(h_ref, h_out);

        double gflops = (2.0 * M * N * K) / 1e9;
        std::printf("[SGEMM 1024x1024x1024]\n");
        std::printf("  Naive:          %7.3f ms | %6.1f GFLOP/s | max_err=%.2e\n", ms_naive, gflops / (ms_naive * 1e-3), err_naive);
        std::printf("  SMEM Tiled:     %7.3f ms | %6.1f GFLOP/s | max_err=%.2e (%.2fx speedup)\n",
                    ms_tiled, gflops / (ms_tiled * 1e-3), err_tiled, ms_naive / ms_tiled);

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    return 0;
}
