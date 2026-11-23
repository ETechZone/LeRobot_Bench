#include "kernels.cuh"
#include "cuda_utils.cuh"
#include <cfloat>
#include <cmath>

// Baseline: 1 thread per row (3 passes over HBM)
__global__ void softmax_naive_kernel(const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * cols;
    float* y_row = y + row * cols;

    float max_val = -FLT_MAX;
    for (int c = 0; c < cols; ++c) max_val = fmaxf(max_val, x_row[c]);

    float denom = 0.0f;
    for (int c = 0; c < cols; ++c) {
        float e = expf(x_row[c] - max_val);
        y_row[c] = e;
        denom += e;
    }
    float inv_denom = 1.0f / denom;
    for (int c = 0; c < cols; ++c) y_row[c] *= inv_denom;
}

struct OnlineSoftmaxState {
    float max_val;
    float sum_exp;
};

__inline__ __device__ OnlineSoftmaxState reduce_online_state(OnlineSoftmaxState a, OnlineSoftmaxState b) {
    if (a.max_val == -FLT_MAX) return b;
    if (b.max_val == -FLT_MAX) return a;
    OnlineSoftmaxState res;
    res.max_val = fmaxf(a.max_val, b.max_val);
    res.sum_exp = a.sum_exp * expf(a.max_val - res.max_val) +
                  b.sum_exp * expf(b.max_val - res.max_val);
    return res;
}

__inline__ __device__ OnlineSoftmaxState warp_reduce_online(OnlineSoftmaxState val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        OnlineSoftmaxState other;
        other.max_val = __shfl_down_sync(0xffffffff, val.max_val, offset);
        other.sum_exp = __shfl_down_sync(0xffffffff, val.sum_exp, offset);
        val = reduce_online_state(val, other);
    }
    return val;
}

// Optimized: Online single-pass max+sum reduction using warp shuffles
__global__ void softmax_online_warp_kernel(const float* __restrict__ x,
                                           float* __restrict__ y,
                                           int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    __shared__ OnlineSoftmaxState smem_warps[32];

    const float* x_row = x + row * cols;
    float* y_row = y + row * cols;

    OnlineSoftmaxState local{-FLT_MAX, 0.0f};
    for (int c = tid; c < cols; c += blockDim.x) {
        OnlineSoftmaxState curr{x_row[c], 1.0f};
        local = reduce_online_state(local, curr);
    }

    local = warp_reduce_online(local);
    if (lane == 0) smem_warps[warp_id] = local;
    __syncthreads();

    OnlineSoftmaxState block_state = (tid < num_warps) ? smem_warps[tid] : OnlineSoftmaxState{-FLT_MAX, 0.0f};
    if (warp_id == 0) {
        block_state = warp_reduce_online(block_state);
        if (tid == 0) smem_warps[0] = block_state;
    }
    __syncthreads();

    float row_max = smem_warps[0].max_val;
    float inv_sum = 1.0f / smem_warps[0].sum_exp;

    for (int c = tid; c < cols; c += blockDim.x) {
        y_row[c] = expf(x_row[c] - row_max) * inv_sum;
    }
}

void launch_softmax_naive(const float* d_x, float* d_y, int rows, int cols, cudaStream_t stream) {
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    softmax_naive_kernel<<<blocks, threads, 0, stream>>>(d_x, d_y, rows, cols);
}

void launch_softmax_online_warp(const float* d_x, float* d_y, int rows, int cols, cudaStream_t stream) {
    int threads = 256;
    softmax_online_warp_kernel<<<rows, threads, 0, stream>>>(d_x, d_y, cols);
}
