#include "kernels.cuh"
#include "cuda_utils.cuh"
#include <cmath>

// Baseline: 1 thread per row (non-coalesced strided memory access)
__global__ void rmsnorm_naive_kernel(const float* __restrict__ x,
                                     const float* __restrict__ weight,
                                     float* __restrict__ y,
                                     int rows, int cols, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * cols;
    float* y_row = y + row * cols;

    float sum_sq = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float val = x_row[c];
        sum_sq += val * val;
    }
    float inv_rms = rsqrtf(sum_sq / static_cast<float>(cols) + eps);
    for (int c = 0; c < cols; ++c) {
        y_row[c] = x_row[c] * inv_rms * weight[c];
    }
}

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized: 1 block per row, vectorized float4 loads + warp-shuffle reduction
__global__ void rmsnorm_warp_vec4_kernel(const float* __restrict__ x,
                                         const float* __restrict__ weight,
                                         float* __restrict__ y,
                                         int cols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    __shared__ float smem_warp_sums[32]; // Max 1024 threads -> 32 warps

    const float* x_row = x + row * cols;
    float* y_row = y + row * cols;

    float local_sum_sq = 0.0f;
    int vec_cols = cols >> 2; // cols / 4
    const float4* x_vec = reinterpret_cast<const float4*>(x_row);

    for (int i = tid; i < vec_cols; i += blockDim.x) {
        float4 v = x_vec[i];
        local_sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int c = (vec_cols << 2) + tid; c < cols; c += blockDim.x) {
        float val = x_row[c];
        local_sum_sq += val * val;
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane == 0) smem_warp_sums[warp_id] = local_sum_sq;
    __syncthreads();

    float block_sum = (tid < num_warps) ? smem_warp_sums[tid] : 0.0f;
    if (warp_id == 0) {
        block_sum = warp_reduce_sum(block_sum);
        if (tid == 0) {
            smem_warp_sums[0] = rsqrtf(block_sum / static_cast<float>(cols) + eps);
        }
    }
    __syncthreads();

    float inv_rms = smem_warp_sums[0];

    const float4* w_vec = reinterpret_cast<const float4*>(weight);
    float4* y_vec = reinterpret_cast<float4*>(y_row);
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        float4 vx = x_vec[i];
        float4 vw = w_vec[i];
        float4 vy;
        vy.x = vx.x * inv_rms * vw.x;
        vy.y = vx.y * inv_rms * vw.y;
        vy.z = vx.z * inv_rms * vw.z;
        vy.w = vx.w * inv_rms * vw.w;
        y_vec[i] = vy;
    }
    for (int c = (vec_cols << 2) + tid; c < cols; c += blockDim.x) {
        y_row[c] = x_row[c] * inv_rms * weight[c];
    }
}

void launch_rmsnorm_naive(const float* d_x, const float* d_w, float* d_y, int rows, int cols, float eps, cudaStream_t stream) {
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    rmsnorm_naive_kernel<<<blocks, threads, 0, stream>>>(d_x, d_w, d_y, rows, cols, eps);
}

void launch_rmsnorm_warp_vec4(const float* d_x, const float* d_w, float* d_y, int rows, int cols, float eps, cudaStream_t stream) {
    int threads = 256;
    rmsnorm_warp_vec4_kernel<<<rows, threads, 0, stream>>>(d_x, d_w, d_y, cols, eps);
}
