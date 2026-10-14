#!/bin/bash
# Gate shader round 3, lease 2, on alight (gpuq run ID -- bash THIS, from ~/trashcan-exp/flashnext). Power levels and
# the kernel log first. Release-always steps first, each gate opened 200 us plus a random 0-1000 us after it started
# (so releases fall at every phase of the poll cycle): strided sweeps near the threshold lease 1 found; then the
# chosen sweep (SWEEP, default 8 MiB read at a 4 KiB stride) in fresh processes (fresh allocations) on both GPUs, with
# write-combined memory and with nontemporal polls. Then never-released steps (bounds 50 and 100 ms) for every new
# shader. Last, the open-gate cost on GPU 1. The kernel log is checked after every step; the first hit stops all.
cd ~/trashcan-exp/flashnext || exit 1
. ./gate3-common.sh
SWEEP=${SWEEP:-"--sweep-kib 8192 --stride 4096"}
power
j="--step release --bound-ms 50 --jitter-us 1000"
for spec in 4096:4096 6144:4096 8192:4096 12288:4096 16384:4096 8192:8192 16384:8192 4096:2048 32768:16384 1024:64; do
    kib=${spec%:*} stride=${spec#*:}
    step "l2-scan-${kib}k-s$stride" $j --reps 200 --shader qwen4_gate_sweep --sweep-kib "$kib" --stride "$stride"
done
for k in 1 2 3 4 5; do
    GPU=0 step "l2-rob-gpu0-$k" $j --reps 200 --shader qwen4_gate_sweep $SWEEP
done
for k in 1 2 3; do
    GPU=1 step "l2-rob-gpu1-$k" $j --reps 200 --shader qwen4_gate_sweep $SWEEP
done
for k in 1 2; do
    GPU=0 step "l2-rob-wc-$k" $j --reps 200 --shader qwen4_gate_sweep $SWEEP --memory uncached --fence sfence
done
GPU=0 step l2-rob-nt-1 $j --reps 200 --shader qwen4_gate_sweep_nt $SWEEP
n="--step never"
GPU=0 step l2-never-atomic-50 $n --bound-ms 50 --shader qwen4_gate_atomic
GPU=0 step l2-never-nosweep-50 $n --bound-ms 50 --shader qwen4_gate_sweep --sweep-kib 0
GPU=0 step l2-never-sweep-gpu0-50 $n --bound-ms 50 --shader qwen4_gate_sweep $SWEEP
GPU=0 step l2-never-sweep-gpu0-100 $n --bound-ms 100 --shader qwen4_gate_sweep $SWEEP
GPU=1 step l2-never-sweep-gpu1-50 $n --bound-ms 50 --shader qwen4_gate_sweep $SWEEP
GPU=1 step l2-never-sweep-gpu1-100 $n --bound-ms 100 --shader qwen4_gate_sweep $SWEEP
GPU=0 step l2-never-sweepnt-50 $n --bound-ms 50 --shader qwen4_gate_sweep_nt $SWEEP
GPU=0 step l2-never-sweep-1m-50 $n --bound-ms 50 --shader qwen4_gate_sweep --sweep-kib 1024 --stride 64
o="--step open --bound-ms 50 --reps 200"
GPU=1 step l2-open-none-gpu1 $o --shader none
GPU=1 step l2-open-sweep-gpu1 $o --shader qwen4_gate_sweep $SWEEP
echo "lease 2 done; kernel log matches $(hits | wc -l) (baseline $base)"
