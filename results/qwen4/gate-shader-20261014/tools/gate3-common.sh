#!/bin/bash
# Shared by the gate-shader round-3 lease scripts (sourced on alight from ~/trashcan-exp/flashnext): the kernel-log
# check, the power-level check, one gate-check step, one machine-code dump. Every step's gate has a bound of
# 50-100 ms. DRY=1 lists the steps and touches neither the GPUs nor the kernel log.
R=results/gate3
bin=./trashcan-gate3
icd=/usr/share/vulkan/icd.d/radeon_icd.json
mkdir -p "$R/clocks"
pat='amdgpu.*(timeout|reset|fault)|CmpltTO|device inaccessible|BUG|Oops'
hits() { sudo -n dmesg | grep -i -E "$pat"; }
# The pattern (case-insensitive) also matches boot messages such as "Kernel Debug File System" and "[Firmware Bug]";
# those are the baseline, listed here. Any new match stops everything.
if [ -z "${DRY:-}" ]; then
    base=$(hits | wc -l)
    echo "kernel log matches before (baseline): $base"
    hits | sed 's/^/    /'
fi
# Stop at the first kernel-log hit and touch nothing more.
check() {
    local now
    [ -n "${DRY:-}" ] && return
    now=$(hits | wc -l)
    if [ "$now" != "$base" ]; then
        echo "STOP after $1: kernel log hit"
        hits | tail -10
        exit 3
    fi
}
# Both GPUs must accept the high level (then back to auto) and report a power level.
power() {
    local c f lvl
    if [ -n "${DRY:-}" ]; then
        echo "DRY power: high then auto on card0 and card1, power level from amdgpu_pm_info"
        return
    fi
    for c in 0 1; do
        f=/sys/class/drm/card$c/device/power_dpm_force_performance_level
        if ! echo high | sudo -n tee "$f" > /dev/null; then
            echo "STOP: card$c refuses high"
            exit 2
        fi
        if ! echo auto | sudo -n tee "$f" > /dev/null; then
            echo "STOP: card$c refuses auto"
            exit 2
        fi
        lvl=$(sudo -n grep "power level" /sys/kernel/debug/dri/$c/amdgpu_pm_info | tr -s ' ' | tr '\n' ';')
        if [ -z "$lvl" ]; then
            echo "STOP: card$c reports no power level"
            exit 2
        fi
        echo "card$c: accepts high, now $(cat "$f"); $lvl"
    done
    check power
}
# step NAME ARGS: one gate-check step on GPU ${GPU:-0}, its JSON line in $R/NAME.json, then the kernel-log check.
step() {
    local name=$1 rc
    shift
    if [ -n "${DRY:-}" ]; then
        echo "DRY $name: $bin qwen4 gate-check --gpu ${GPU:-0} $*"
        return
    fi
    TRASHCAN_TP_PROFILE=1 scripts/with-clocks.sh "$R/clocks/$name.txt" "$bin" qwen4 gate-check --gpu "${GPU:-0}" "$@" \
        > "$R/$name.json" 2> "$R/$name.err"
    rc=$?
    echo "$name: rc $rc; $(cut -c1-400 "$R/$name.json")"
    check "$name"
    if [ "$rc" != 0 ]; then
        echo "STOP after $name: rc $rc; $(tail -3 "$R/$name.err")"
        exit 4
    fi
}
# isa NAME [ENV...]: the gate shaders' machine code (gate-isa on GPU ${GPU:-0}, or RADV's null device with
# RADV_FORCE_FAMILY in ENV), then the kernel-log check.
isa() {
    local name=$1 rc
    shift
    if [ -n "${DRY:-}" ]; then
        echo "DRY $name: env $* VK_ICD_FILENAMES=$icd $bin qwen4 gate-isa --gpu ${GPU:-0}"
        return
    fi
    env "$@" VK_ICD_FILENAMES=$icd "$bin" qwen4 gate-isa --gpu "${GPU:-0}" > "$R/$name.txt" 2> "$R/$name.err"
    rc=$?
    echo "$name: rc $rc, $(wc -l < "$R/$name.txt") lines"
    check "$name"
}
