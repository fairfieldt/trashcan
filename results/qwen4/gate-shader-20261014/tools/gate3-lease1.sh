#!/bin/bash
# Gate shader round 3, lease 1, on alight GPU 0 (gpuq run ID -- bash THIS, from ~/trashcan-exp/flashnext).
# Power levels and the kernel log first; the gate shaders' machine code on GPU 0 and on RADV's null device; then
# release-always steps only (the host opens every gate 200 us after it starts; bound 50 ms, so a gate that never sees
# the write ends after 50 ms): test 1 (device-scope atomic loads), test 3 (host fences alone), test 2 (an L2 sweep
# between polls: sizes and strides), tests 2+3 together; last, the cost of an already-open gate. No never-released
# step here. The kernel log is checked after every step; the first hit stops everything.
cd ~/trashcan-exp/flashnext || exit 1
. ./gate3-common.sh
power
isa isa-gpu0
isa isa-null-tahiti RADV_FORCE_FAMILY=tahiti
r="--step release --bound-ms 50"
# Test 1: device-scope atomic loads (expected to behave like volatile loads).
step t1-atomic-cached-mfence $r --reps 20 --max-fail 1 --shader qwen4_gate_atomic --memory cached --fence mfence
step t1-atomic-wc-sfence $r --reps 20 --max-fail 1 --shader qwen4_gate_atomic --memory uncached --fence sfence
# Test 3: host fences alone (no sweep).
step t3-volatile-cached-clflush $r --reps 20 --max-fail 1 --shader qwen4_gate_clock --memory cached --fence clflush
step t3-volatile-wc-sfence $r --reps 20 --max-fail 1 --shader qwen4_gate_clock --memory uncached --fence sfence
step t3-nt-cached-clflush $r --reps 20 --max-fail 1 --shader qwen4_gate_nt --memory cached --fence clflush
step t3-nt-wc-sfence $r --reps 20 --max-fail 1 --shader qwen4_gate_nt --memory uncached --fence sfence
# Test 2: L2 sweep between polls (volatile polls, host-cached memory, mfence). No sweep first (one wave, timed polls).
step t2-nosweep $r --reps 20 --max-fail 1 --shader qwen4_gate_sweep --sweep-kib 0
for kib in 256 512 768 1024 2048 4096; do
    step "t2-sweep-${kib}k-s64" $r --reps 100 --shader qwen4_gate_sweep --sweep-kib $kib --stride 64
done
for spec in 1024:256 4096:256 4096:1024 2048:4096 8192:4096 32768:4096; do
    kib=${spec%:*} stride=${spec#*:}
    step "t2-sweep-${kib}k-s$stride" $r --reps 100 --shader qwen4_gate_sweep --sweep-kib "$kib" --stride "$stride"
done
# Tests 2 and 3 together (1 MiB sweep): write-combined memory with sfence, cached with clflush; nontemporal polls.
step t23-sweep-wc-sfence $r --reps 100 --shader qwen4_gate_sweep --sweep-kib 1024 --memory uncached --fence sfence
step t23-sweep-cached-clflush $r --reps 100 --shader qwen4_gate_sweep --sweep-kib 1024 --memory cached --fence clflush
step t23-sweepnt-cached-mfence $r --reps 100 --shader qwen4_gate_sweep_nt --sweep-kib 1024 --memory cached --fence mfence
step t23-sweepnt-wc-sfence $r --reps 100 --shader qwen4_gate_sweep_nt --sweep-kib 1024 --memory uncached --fence sfence
# An already-open gate: no gate, the plain volatile gate, the sweep gate (its first poll reads 1, so it never sweeps).
o="--step open --bound-ms 50 --reps 200"
step open-none $o --shader none
step open-volatile $o --shader qwen4_gate_clock
step open-sweep $o --shader qwen4_gate_sweep --sweep-kib 1024
echo "lease 1 done; kernel log matches $(hits | wc -l) (baseline $base)"
