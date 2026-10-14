#!/bin/bash
# Round 2 of the bounded gate's single-GPU check, run on alight under a gpuq lease (gpuq run ID -- bash THIS):
# both GPUs' power level first (each must accept `high`), then on GPU 0 variant (b), the gate word in host-visible,
# coherent, non-cached memory with volatile loads, then variant (a), nontemporal loads on host-cached memory. For each:
# release-always (200 runs, bound 100 ms), then never released (bounds 50 and 100 ms). The kernel log is checked after
# every step; the first amdgpu timeout or reset stops everything.
cd ~/trashcan-exp/flashnext || exit 1
out=results/host-gate2-alight.json
: > "$out"
errors() { sudo -n dmesg | grep -c -i -E 'amdgpu.*(timeout|reset)|gpu reset'; }
base=$(errors)
echo "amdgpu timeouts/resets before: $base"
for c in 0 1; do
    f=/sys/class/drm/card$c/device/power_dpm_force_performance_level
    echo "card$c: level $(cat $f), state $(cat /sys/class/drm/card$c/device/power_dpm_state)"
    if ! echo high | sudo -n tee "$f" > /dev/null 2>&1; then
        echo "card$c refuses the high level: stop"
        exit 2
    fi
    echo auto | sudo -n tee "$f" > /dev/null
done
echo "power levels: both GPUs accept high"
step() {
    TRASHCAN_TP_PROFILE=1 scripts/with-clocks.sh "results/clocks/host-gate2-alight-$1.txt" ./trashcan-host-gate2 qwen4 gate-check \
        --gpu 0 "${@:2}" >> "$out" 2> "results/host-gate2-alight-$1.err"
    local rc=$? now
    now=$(errors)
    echo "$1: rc $rc, amdgpu timeouts/resets $now; $(tail -1 "$out" | cut -c1-400)"
    if [ "$now" != "$base" ]; then
        echo "STOP: an amdgpu timeout or reset after $1"
        exit 3
    fi
}
step b-release --step release --memory uncached --shader qwen4_gate_clock --bound-ms 100 --reps 200
step b-never-50 --step never --memory uncached --shader qwen4_gate_clock --bound-ms 50
step b-never-100 --step never --memory uncached --shader qwen4_gate_clock --bound-ms 100
step a-release --step release --memory cached --shader qwen4_gate_nt --bound-ms 100 --reps 200
step a-never-50 --step never --memory cached --shader qwen4_gate_nt --bound-ms 50
step a-never-100 --step never --memory cached --shader qwen4_gate_nt --bound-ms 100
echo "gate check round 2 done"
