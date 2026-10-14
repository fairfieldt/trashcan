#!/bin/bash
# Clock levels with two engine processes on one Mac (gpuq run ID -- bash THIS, from ~/trashcan-exp/flashnext): for
# main's binary (A) and the fix (B), `stage-bench` on GPU 1 (long: layers 3-5, 1500 decode steps) and, started with
# it, `stage-bench` on GPU 0 (short: layers 0-2, 100 decode steps), while both GPUs' DPM level files are sampled every
# 50 ms. When the short process ends, its clock holder returns its GPUs to auto: with main both GPUs, with the fix only
# GPU 0, so GPU 1 must stay high until its own process ends. The journal's sudo lines show the level files each write
# touched. The kernel log (amdgpu faults, PCIe AER) is checked after each binary; a process starts only with the VRAM
# it loads free on its GPU. DRY=1 lists the steps.
cd ~/trashcan-exp/flashnext || exit 1
R=${R:-results/clk}
mkdir -p "$R"
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
A=${A:-./trashcan-dgbase3}
B=${B:-./trashcan-clk1}
C="--context 4096 --batch 128 --prefill 1"
pat='amdgpu.*(timeout|reset|fault)|CmpltTO|device inaccessible|BUG|Oops|AER|PCIe Bus Error|pci_channel_io|UnxCmplt'
hits() { sudo -n dmesg | grep -i -E "$pat"; }
trap 'kill $(jobs -p) 2>/dev/null; wait 2>/dev/null' EXIT
if [ -z "${DRY:-}" ]; then
    base=$(hits | wc -l)
    echo "kernel $(uname -r); kernel log matches before (baseline): $base"
    a=$(basename "$(readlink -f /sys/class/drm/card0/device)")
    b=$(basename "$(readlink -f /sys/class/drm/card1/device)")
    [[ "$a" < "$b" ]] || { echo "STOP: card0 ($a) does not come before card1 ($b) on the PCI bus"; exit 2; }
fi
check() {
    [ -n "${DRY:-}" ] && return
    if [ "$(hits | wc -l)" != "$base" ]; then
        echo "STOP after $1: kernel log hit"
        hits | tail -12
        exit 3
    fi
}
lvl() { cat /sys/class/drm/card$1/device/power_dpm_force_performance_level; }
free_mib() {
    local f=/sys/class/drm/card$1/device
    echo $((($(cat $f/mem_info_vram_total) - $(cat $f/mem_info_vram_used)) >> 20))
}
need_vram() {
    local i
    for i in $(seq 100); do
        [ "$(free_mib "$1")" -ge 4800 ] && return
        sleep 0.1
    done
    echo "STOP: GPU $1 has $(free_mib "$1") MiB of VRAM free"
    exit 6
}
run() {
    local name=$1 bin=$2 long short since
    if [ -n "${DRY:-}" ]; then
        echo "DRY $name: $bin stage-bench --ranges 3:6 --gpus 1 --decode 1500 and --ranges 0:3 --gpus 0 --decode 100"
        return
    fi
    need_vram 0
    need_vram 1
    since=$(date '+%Y-%m-%d %H:%M:%S')
    sleep 1
    $bin qwen4 stage-bench $M --ranges 3:6 --gpus 1 $C --decode 1500 > "$R/$name-long.out" 2> "$R/$name-long.err" &
    long=$!
    $bin qwen4 stage-bench $M --ranges 0:3 --gpus 0 $C --decode 100 > "$R/$name-short.out" 2> "$R/$name-short.err" &
    short=$!
    while kill -0 $long 2>/dev/null; do
        echo "$(date +%s.%N | cut -c1-14) $(lvl 0) $(lvl 1) $(kill -0 $short 2>/dev/null && echo short-running || echo short-ended)"
        sleep 0.05
    done > "$R/$name-levels.txt"
    wait
    sleep 1
    check "$name"
    journalctl --since "$since" --no-pager | grep "COMMAND=/usr/bin/tee" | sed 's/.*COMMAND=//' > "$R/$name-tee.txt"
    echo "$name: level writes: $(wc -l < "$R/$name-tee.txt"), of them naming both GPUs: \
$(grep -c "card0.*card1" "$R/$name-tee.txt")"
    echo "$name: GPU 1's level in the samples after the short process ended: \
$(awk '$4 == "short-ended" { print $3 }' "$R/$name-levels.txt" | sort | uniq -c | tr '\n' ' ')"
}
# Both GPUs must accept the high level (then back to auto) before anything runs.
for c in 0 1; do
    [ -n "${DRY:-}" ] && break
    echo high | sudo -n tee /sys/class/drm/card$c/device/power_dpm_force_performance_level > /dev/null || exit 2
    echo auto | sudo -n tee /sys/class/drm/card$c/device/power_dpm_force_performance_level > /dev/null || exit 2
done
run A "$A"
run B "$B"
if pgrep -af 'trashcan-(dgbase3|clk1) qwen4'; then
    pkill -9 -f 'trashcan-(dgbase3|clk1) qwen4'
fi
echo "levels now: GPU 0 $(lvl 0), GPU 1 $(lvl 1); kernel log matches $(hits | wc -l) (baseline ${base:-})"
