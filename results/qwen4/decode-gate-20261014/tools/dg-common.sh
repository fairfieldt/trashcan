#!/bin/bash
# Shared by the decode-gate lease scripts (sourced on the Mac from ~/trashcan-exp/flashnext): the power-level check,
# the kernel-log check after every step (amdgpu faults and PCIe AER), the free-VRAM check before every engine process,
# one logged step. DRY=1 lists the steps and touches neither the GPUs nor the kernel log.
R=${R:-results/dg}
mkdir -p "$R/clocks"
pat='amdgpu.*(timeout|reset|fault)|CmpltTO|device inaccessible|BUG|Oops|AER|PCIe Bus Error|pci_channel_io|UnxCmplt'
hits() { sudo -n dmesg | grep -i -E "$pat"; }
# The pattern (case-insensitive) also matches boot messages such as "Kernel Debug File System", "[Firmware Bug]" and
# AER's own setup lines: they are the baseline, listed here. Any new match stops everything.
if [ -z "${DRY:-}" ]; then
    base=$(hits | wc -l)
    echo "kernel log matches before (baseline): $base"
    hits | sed 's/^/    /'
fi
check() {
    local now
    [ -n "${DRY:-}" ] && return
    now=$(hits | wc -l)
    if [ "$now" != "$base" ]; then
        echo "STOP after $1: kernel log hit"
        hits | tail -12
        exit 3
    fi
}
# The engine numbers the AMD GPUs by PCI bus; these scripts take engine GPU N to be cardN.
if [ -z "${DRY:-}" ]; then
    a=$(basename "$(readlink -f /sys/class/drm/card0/device)")
    b=$(basename "$(readlink -f /sys/class/drm/card1/device)")
    if [[ ! "$a" < "$b" ]]; then
        echo "STOP: card0 ($a) does not come before card1 ($b) on the PCI bus"
        exit 2
    fi
    echo "GPU 0 is card0 ($a), GPU 1 is card1 ($b)"
fi
# need_vram GPU MIB: stop unless GPU has MIB MiB of VRAM free (amdgpu's counters), waiting up to 10 s for a process
# that just ended to free its memory. No engine process may share a GPU's VRAM with another (on 2026-10-14 two
# processes overcommitting alight's GPU 1 preceded a PCIe completion timeout and the loss of the GPU).
need_vram() {
    local f=/sys/class/drm/card$1/device free i
    if [ -n "${DRY:-}" ]; then
        echo "DRY need $2 MiB of VRAM free on GPU $1"
        return
    fi
    for i in $(seq 100); do
        free=$((($(cat $f/mem_info_vram_total) - $(cat $f/mem_info_vram_used)) >> 20))
        [ "$free" -ge "$2" ] && return
        sleep 0.1
    done
    echo "STOP: GPU $1 has $free MiB of VRAM free; the process needs $2"
    exit 6
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
        echo high | sudo -n tee "$f" > /dev/null || { echo "STOP: card$c refuses high"; exit 2; }
        echo auto | sudo -n tee "$f" > /dev/null || { echo "STOP: card$c refuses auto"; exit 2; }
        lvl=$(sudo -n grep "power level" /sys/kernel/debug/dri/$c/amdgpu_pm_info | tr -s ' ' | tr '\n' ';')
        [ -n "$lvl" ] || { echo "STOP: card$c reports no power level"; exit 2; }
        echo "card$c: accepts high, now $(cat "$f"); $lvl"
    done
    check power
}
# step NAME CMD...: CMD under with-clocks, stdout to $R/NAME.out, stderr to $R/NAME.err, then the kernel-log check.
# A nonzero exit stops everything unless ALLOW_FAIL=1.
step() {
    local name=$1 rc
    shift
    if [ -n "${DRY:-}" ]; then
        echo "DRY $name: $*"
        return
    fi
    scripts/with-clocks.sh "$R/clocks/$name.txt" "$@" > "$R/$name.out" 2> "$R/$name.err"
    rc=$?
    echo "$name: rc $rc; $(tail -1 "$R/$name.out" | cut -c1-400)"
    check "$name"
    if [ "$rc" != 0 ] && [ -z "${ALLOW_FAIL:-}" ]; then
        echo "STOP after $name: rc $rc; $(tail -3 "$R/$name.err")"
        exit 4
    fi
}
# sweeps LOG...: stop unless every gate check in the logs timed its sweeps at VRAM speed (2-20 shader-clock ticks per
# line; the gate gives up above 32): otherwise its cutoff for sweeps of evicted memory would not work.
sweeps() {
    local bad
    [ -n "${DRY:-}" ] && return
    bad=$(grep -h "check, .* per line" "$@" | grep -o -E "\([0-9]+ per line\)" | tr -d '(' | \
        awk '$1 < 2 || $1 > 20' | head -3)
    if [ -n "$bad" ]; then
        echo "STOP: a gate check timed its sweep at $bad ticks per line"
        exit 7
    fi
    # Gates that went off for the process, and stages with gates (a test of the gates needs some).
    grep -h "no longer submitted ahead\|not submitted ahead" "$@" | sed 's/^/    /'
    echo "    stages with gates: $(cat "$@" 2>/dev/null | grep -c "gates read")"
}
# No engine process may outlive a script.
leftovers() {
    [ -n "${DRY:-}" ] && return
    sleep 1
    if pgrep -af 'trashcan-dg[a-z0-9]* qwen4'; then
        echo "engine processes left: killing them"
        pkill -9 -f 'trashcan-dg[a-z0-9]* qwen4'
    fi
}
# The VRAM each engine process of these scripts loads (6 layers of Flash-Next and half the head: 4634-4664 MiB), with
# room for the driver's own allocations.
NEED=4800
