#!/bin/bash
# Gate shader round 3, lease 3, on alight (gpuq run ID -- bash THIS, from ~/trashcan-exp/flashnext). Every gate's
# shader-clock bound is now counted at 300 MHz, the lowest DPM level (s_memtime counts cycles of the current shader
# clock on GFX6; lease 2 stopped when GPU 1 sat at 300 MHz), so it holds at every level; every run must also end
# within its bound by the command processor's clock. Power levels and the kernel log first. Release-always first:
# GPU 0 forced to the low DPM level (300/150 MHz), the worst case. Then never-released steps (bounds 50 and 100 ms),
# at the low level and then with both GPUs held high as the engine holds them. Then the open-gate cost on GPU 1, and
# GPU 0's power while gates wait (SMC PG0R): idle, then 15 s of never-released gates back to back for a plain poll,
# the chosen sweep and a 1 MiB contiguous sweep, then idle again. The levels go back to auto on any exit.
cd ~/trashcan-exp/flashnext || exit 1
. ./gate3-common.sh
SWEEP=${SWEEP:-"--sweep-kib 8192 --stride 4096"}
power
level() {
    local c
    for c in $2; do
        if [ -n "${DRY:-}" ]; then
            echo "DRY card$c level $1"
            continue
        fi
        echo "$1" | sudo -n tee /sys/class/drm/card$c/device/power_dpm_force_performance_level > /dev/null
        echo "card$c level $(cat /sys/class/drm/card$c/device/power_dpm_force_performance_level)"
    done
}
idle() {
    echo "$1: idle from $(date -u +%s.%N)"
    [ -z "${DRY:-}" ] && sleep "$2"
    echo "$1: idle to $(date -u +%s.%N)"
}
trap 'level auto "0 1"' EXIT
# The low level: one release-always step, then the never-released ones.
level low 0
GPU=0 step l3-low-release --step release --bound-ms 50 --jitter-us 1000 --reps 100 --shader qwen4_gate_sweep $SWEEP
GPU=0 step l3-low-never-sweep-50 --step never --bound-ms 50 --shader qwen4_gate_sweep $SWEEP
GPU=0 step l3-low-never-nosweep-50 --step never --bound-ms 50 --shader qwen4_gate_sweep --sweep-kib 0
GPU=0 step l3-low-never-clock-50 --step never --bound-ms 50 --shader qwen4_gate_clock
# Both GPUs held high.
level high "0 1"
n="--step never"
GPU=1 step l3-never-sweep-gpu1-50 $n --bound-ms 50 --shader qwen4_gate_sweep $SWEEP
GPU=1 step l3-never-sweep-gpu1-100 $n --bound-ms 100 --shader qwen4_gate_sweep $SWEEP
GPU=0 step l3-never-sweep-gpu0-100 $n --bound-ms 100 --shader qwen4_gate_sweep $SWEEP
GPU=0 step l3-never-sweepnt-50 $n --bound-ms 50 --shader qwen4_gate_sweep_nt $SWEEP
GPU=0 step l3-never-sweep-1m-50 $n --bound-ms 50 --shader qwen4_gate_sweep --sweep-kib 1024 --stride 64
GPU=1 step l3-never-atomic-gpu1-50 $n --bound-ms 50 --shader qwen4_gate_atomic
o="--step open --bound-ms 50 --reps 200"
GPU=1 step l3-open-none-gpu1 $o --shader none
GPU=1 step l3-open-sweep-gpu1 $o --shader qwen4_gate_sweep $SWEEP
# Power: GPU 1 back to auto (idle), GPU 0 held high.
level auto 1
idle l3-power-idle-1 15
b="--step busy --bound-ms 100 --reps 430"
GPU=0 step l3-power-nosweep $b --shader qwen4_gate_sweep --sweep-kib 0
GPU=0 step l3-power-sweep $b --shader qwen4_gate_sweep $SWEEP
GPU=0 step l3-power-sweep-1m $b --shader qwen4_gate_sweep --sweep-kib 1024 --stride 64
idle l3-power-idle-2 10
level auto "0 1"
echo "lease 3 done; kernel log matches $(hits | wc -l) (baseline $base)"
