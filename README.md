# LeRobot_Bench
Empirical benchmarking suite for Vision-Language-Action (VLA) models on the LeRobot SO-ARM 101 platform—evaluating vision backbones (SigLIP vs. CLIP in SmolVLA), spatial patching tradeoffs, and edge inference latency.

# cuda-llm-kernel-bench

High-performance **CUDA C++** kernel suite and host-side C++17 benchmarking harness targeting core **Large Language Model (LLM)** training and inference primitives.

Designed to demonstrate systematic, hardware-aware GPU optimization: progressing from naive global-memory baselines to vectorized, shared-memory tiled, and warp-primitive (`__shfl_down_sync`) implementations with strict numerical verification against CPU/cuBLAS references.

## Implemented Kernels & Optimization Progression

| Operator | Optimization Stages | Key Techniques Demonstrated |
| :--- | :--- | :--- |
| **Fused RMSNorm** | `v1_naive` -> `v2_shared_mem` -> `v3_warp_shuffle_vec4` | `float4` vectorized global loads, block-level shared memory reduction, warp-level `__shfl_down_sync` (eliminating HBM roundtrips) |
| **Online Safe Softmax** | `v1_two_pass` -> `v2_online_warp_tiled` | Numerical stability (`max` subtraction), single-pass online normalizer, bank-conflict-free shared memory layout |
| **Tiled SGEMM** | `v1_naive` -> `v2_smem_tiled` -> `v3_reg_blocked_vec4` | 2D shared memory tiling (`32x32`), `+1` padding against bank conflicts, register blocking, comparison vs. `cuBLAS` |

## Repository Structure

```text
cuda-llm-kernel-bench/
├── CMakeLists.txt
├── include/
│   ├── cuda_utils.cuh          # Error checking, CUDA event timers, roofline bandwidth calculators
│   └── kernels.cuh             # Unified C++ host-callable kernel interfaces
├── src/
│   ├── main.cu                 # Host-side C++17 runner: correctness checks + latency/GB/s benchmarks
│   └── kernels/
│       ├── rmsnorm.cu          # Naive, Shared Memory, and Vectorized Warp-Shuffle RMSNorm
│       ├── softmax.cu          # Two-Pass vs. Online Warp-Reduced Row-wise Softmax
│       └── sgemm.cu            # Naive, Shared-Memory Tiled, and Register-Blocked Matrix Multiply
└── benchmarks/
    └── profile_nsight.sh       # Reproducible Nsight Compute (ncu) & Nsight Systems (nsys) script
```

## Build & Run

### Prerequisites
* NVIDIA GPU (Compute Capability `sm_75` / `sm_80` / `sm_86` / `sm_89` / `sm_90`)
* CUDA Toolkit 11.8+ or 12.x (`nvcc`)
* CMake 3.18+ & C++17 compiler

### Compilation & Execution
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./llm_kernel_bench
```

### Profiling with Nsight Compute (`ncu`)
```bash
bash benchmarks/profile_nsight.sh ./build/llm_kernel_bench
```

## Verification & Roofline Methodology
1. **Numerical Accuracy:** Every optimized kernel is validated against a double-accumulated CPU reference implementation (`max_abs_err < 1e-5` for FP32).
2. **Hardware Metrics:** Benchmarks measure steady-state kernel execution via `cudaEventRecord` (excluding host-device allocation overhead) and report **Effective Memory Bandwidth (GB/s)** for memory-bound ops (RMSNorm, Softmax) and **TFLOP/s** for compute-bound ops (SGEMM).
