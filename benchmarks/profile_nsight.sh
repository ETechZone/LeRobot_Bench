#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-./build/llm_kernel_bench}"

if ! command -v ncu &> /dev/null; then
    echo "Nsight Compute (ncu) not found in PATH. Ensure CUDA Toolkit profiling tools are installed."
    exit 1
fi

echo "Running Nsight Compute memory & warp stall metrics..."
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct \
    --target-processes all "${BIN}"
