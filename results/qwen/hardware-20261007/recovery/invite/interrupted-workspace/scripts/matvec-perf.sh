#!/bin/bash
# Single-GPU mat-vec throughput: test-backend-ops perf for MUL_MAT on one D700, with clock logging.
# For a memory-bound mat-vec the useful number is bytes of weights read per second. test-backend-ops reports it
# as GB/s for ops with known memory traffic.
# usage (on incite): matvec-perf.sh LABEL [backend, default Vulkan0]
set -u
label=$1
backend=${2:-Vulkan0}
BIN=${BIN:-$HOME/trashcan/llama.cpp/build-tp/bin}
out=$HOME/trashcan/results/$(date +%Y%m%d-%H%M)-matvec-$label
mkdir -p "$out"
"$HOME/trashcan/scripts/clockmon.sh" "$out/clocks.txt" &
cm=$!
trap 'kill $cm' EXIT
"$BIN/test-backend-ops" perf -o MUL_MAT -b "$backend" > "$out/mul_mat.txt" 2>&1
kill $cm
trap - EXIT
grep -E "type_a=(q4_0|q8_0|f16|q4_K),type_b=f32,m=[0-9]+,n=1," "$out/mul_mat.txt" | sed -E 's/MUL_MAT\(//; s/,bs=.*\)//' | tee "$out/summary.txt"
"$HOME/trashcan/scripts/clocksummary.sh" "$out/clocks.txt"
echo "results in $out"
