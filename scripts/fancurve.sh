#!/bin/bash
# Mac Pro 6,1 main fan: full speed while the GPUs work, slower when the Mac is idle and cool.
#
# Full speed is FAN_RPM (default 2350) or the live SMC maximum, whichever is higher. The SMC's maximum (fan1_max) is
# 1900 at boot and drifts between 1900 and ~2380 afterwards; the SMC accepts a manual target above it. FAN_RPM=0
# follows the SMC maximum alone.
#
# The fan runs at full speed while any amdgpu GPU works (its compute fence counters in debugfs advance, or its DPM
# level is held at `high`, as the engine does while work arrives) and for HOLD seconds after. Otherwise the hottest GPU and the
# CPU package set it: IDLE_RPM up to GPU_COOL / CPU_COOL degrees, full speed from GPU_HOT / CPU_HOT, linear in between.
# It rises at once and falls at most STEP rpm per poll. A temperature it cannot read counts as hot.
#
# fan1_max can change after boot, so never cache a successful reading indefinitely.
# Stop/failure returns control to the SMC. Installed as /usr/local/sbin/fancurve.sh.
set -euo pipefail
INTERVAL=${INTERVAL:-2}
FAN_RPM=${FAN_RPM:-2350}
IDLE_RPM=1200 GPU_COOL=50 GPU_HOT=65 CPU_COOL=60 CPU_HOT=80 HOLD=120 STEP=100
smc=${SMC_PATH:-/sys/devices/platform/applesmc.768}
drm=${DRM_PATH:-/sys/class/drm}
dri=${DRI_DEBUG_PATH:-/sys/kernel/debug/dri}
hwmon=${HWMON_PATH:-/sys/class/hwmon}
# Before Linux 7.3 applesmc keeps the fan files in its device directory, with manual control as fan1_manual (1 on,
# 0 off) and the target as fan1_output. From 7.3 they are hwmon attributes in hwmon/hwmonN: pwm1_enable (1 manual,
# 2 automatic) and fan1_target.
if [[ -e $smc/fan1_manual ]]; then
    fan=$smc mode=fan1_manual on=1 off=0 out=fan1_output
else
    fan=$(echo "$smc"/hwmon/hwmon*) mode=pwm1_enable on=1 off=2 out=fan1_target
fi

restore() {
    local status=$?
    trap - EXIT
    if printf '%s\n' "$off" > "$fan/$mode"; then
        echo 'fan returned to automatic SMC control'
    else
        echo 'cannot return fan to automatic SMC control' >&2
        status=1
    fi
    exit "$status"
}
trap restore EXIT
trap 'exit 0' TERM INT

fail() { echo "fan control: $*" >&2; exit 1; }
# Return 1 only for a failed read; malformed values are not transient read errors.
read_rpm() {
    local value
    value=$(cat "$1" 2>/dev/null) || return 1
    [[ $value =~ ^[0-9]{1,9}$ ]] || return 2
    printf '%s\n' "$((10#$value))"
}
# Whole degrees from a millidegree file; empty when it cannot be read.
read_temp() {
    local value
    value=$(cat "$1" 2>/dev/null) || return 0
    [[ $value =~ ^-?[0-9]{1,7}$ ]] && printf '%s\n' "$((value / 1000))"
}

[[ $INTERVAL =~ ^([1-9]|10)$ ]] || fail 'INTERVAL must be 1..10 seconds'
# 2400 rpm caps typos; the SMC's own maximum has reached 2382.
[[ $FAN_RPM =~ ^(0|[1-9][0-9]{0,3})$ ]] && (( FAN_RPM <= 2400 )) || fail 'FAN_RPM must be 0..2400'
minimum=$(read_rpm "$fan/fan1_min") || fail 'cannot read a valid SMC minimum'
(( minimum > 0 )) || fail 'invalid SMC minimum'

# The amdgpu GPUs (cardN whose driver is amdgpu), with their temperature, DPM level and fence files, and the CPU
# package temperature (coretemp's temp1). At boot this service can start before amdgpu has bound the GPUs (5 s against
# 12-18 s on the Macs), so while either is missing they are looked for again on every poll, at full speed meanwhile.
# Once found, a file that disappears later reads as hot.
discover() {
    gpu_temps=() gpu_levels=() gpu_fences=() cpu_temp=
    local card temps h
    for card in "$drm"/card*; do
        [[ ${card##*/} =~ ^card([0-9]+)$ ]] || continue
        [[ $(readlink "$card/device/driver" 2>/dev/null) == */amdgpu ]] || continue
        temps=("$card"/device/hwmon/hwmon*/temp1_input)
        gpu_temps+=("${temps[0]}")
        gpu_levels+=("$card/device/power_dpm_force_performance_level")
        gpu_fences+=("$dri/${BASH_REMATCH[1]}/amdgpu_fence_info")
    done
    for h in "$hwmon"/hwmon*; do
        [[ $(cat "$h/name" 2>/dev/null) == coretemp ]] && cpu_temp=$h/temp1_input && break
    done
    return 0
}
found() { (( ${#gpu_temps[@]} > 0 )) && [[ -n $cpu_temp ]]; }
discover
(( ${#gpu_temps[@]} > 0 )) || echo 'no amdgpu GPU yet: full speed until one appears' >&2
[[ -n $cpu_temp ]] || echo 'no coretemp sensor yet: full speed until it appears' >&2

# The sum of the compute rings' last signaled fences over all GPUs (`--- ring N (comp_...) ---`, as the port lights
# count them); it changes whenever a GPU finishes a compute job. trashcan's work is compute; a desktop drawing on a GPU
# signals the gfx and sdma rings and does not count. Empty when debugfs cannot be read (then only the DPM level and the
# temperatures count).
fences() {
    local f a b c d compute sum=0
    (( ${#gpu_fences[@]} > 0 )) || return 0
    for f in "${gpu_fences[@]}"; do
        [[ -r $f ]] || return 0
        compute=0
        while read -r a b c d; do
            if [[ $a == --- ]]; then [[ $d == '(comp_'* ]] && compute=1 || compute=0
            elif (( compute )) && [[ "$a $b $c" == 'Last signaled fence' && $d =~ ^0x[0-9a-fA-F]+$ ]]; then
                sum=$((sum + 16#${d#0x}))
            fi
        done < "$f"
    done
    printf '%s\n' "$sum"
}
# Sets busy=1 when a GPU works: its fences advanced since the last poll or its DPM level is held at `high`. Runs in
# this shell (not $(...)) so that last_fences carries over.
last_fences=$(fences)
check_work() {
    local f now
    now=$(fences) busy=0
    [[ -n $now && $now != "$last_fences" ]] && busy=1
    last_fences=$now
    (( ${#gpu_levels[@]} > 0 )) || return 0
    for f in "${gpu_levels[@]}"; do
        [[ $(cat "$f" 2>/dev/null) == high ]] && busy=1
    done
    return 0
}
# Per mille of the way from IDLE_RPM to full speed that the temperatures ask for: 1000 for a sensor that cannot be read.
need() {
    local f t n worst=0
    (( ${#gpu_temps[@]} > 0 )) && [[ -n $cpu_temp ]] || { echo 1000; return; }
    for f in "${gpu_temps[@]}"; do
        t=$(read_temp "$f"); [[ -n $t ]] || { echo 1000; return; }
        n=$(( (t - GPU_COOL) * 1000 / (GPU_HOT - GPU_COOL) )); (( n > worst )) && worst=$n
    done
    t=$(read_temp "$cpu_temp"); [[ -n $t ]] || { echo 1000; return; }
    n=$(( (t - CPU_COOL) * 1000 / (CPU_HOT - CPU_COOL) )); (( n > worst )) && worst=$n
    echo $(( worst > 1000 ? 1000 : worst ))
}

last_max= failures=0 target= logged_policy= logged_target=0 quiet=$(( HOLD / INTERVAL ))
echo "fan policy: full speed ($FAN_RPM rpm or the live SMC maximum if higher) while a GPU works and ${HOLD}s after;" \
     "otherwise $IDLE_RPM rpm up to ${GPU_COOL}C GPU / ${CPU_COOL}C CPU, full speed from ${GPU_HOT}C / ${CPU_HOT}C;" \
     "refresh every ${INTERVAL}s"
while :; do
    if maximum=$(read_rpm "$fan/fan1_max"); then
        (( maximum >= minimum )) || fail "SMC maximum $maximum is below minimum $minimum"
        if (( failures > 0 )); then echo 'SMC maximum readable again'; fi
        failures=0
        last_max=$maximum
    else
        read_status=$?
        (( read_status == 1 )) || fail 'invalid SMC maximum value'
        failures=$((failures + 1))
        [[ -n $last_max ]] || fail 'SMC maximum unavailable at startup'
        (( failures < 3 )) || fail 'SMC maximum unreadable for three polls'
        if (( failures == 1 )); then
            echo "SMC maximum temporarily unreadable; retaining last valid target $last_max rpm" >&2
        fi
        maximum=$last_max
    fi
    full=$(( maximum > FAN_RPM ? maximum : FAN_RPM ))
    idle=$(( IDLE_RPM > minimum ? IDLE_RPM : minimum ))
    (( idle < full )) || idle=$full
    if ! found; then
        discover
        if found; then
            echo "found ${#gpu_temps[@]} amdgpu GPU(s) and the CPU sensor"
            last_fences=$(fences)
        fi
    fi
    check_work
    if (( busy )); then quiet=0; else quiet=$((quiet + 1)); fi
    if (( quiet * INTERVAL < HOLD )); then
        policy=work want=$full
    else
        n=$(need)
        policy=temperature want=$(( idle + (full - idle) * (n < 0 ? 0 : n) / 1000 ))
    fi
    # Up at once, down from full speed in steps (full speed itself follows the SMC maximum).
    if [[ -z $target ]] || (( want >= target || want == full )); then target=$want
    else target=$(( want > target - STEP ? want : target - STEP )); fi
    manual=$(cat "$fan/$mode") || fail 'cannot read fan control mode'
    [[ $manual == "$off" || $manual == "$on" ]] || fail 'invalid fan control mode'
    output=$(read_rpm "$fan/$out") || fail 'cannot read fan target'
    if [[ $manual != "$on" ]]; then printf '%s\n' "$on" > "$fan/$mode"; fi
    if [[ $manual != "$on" || $output != "$target" ]]; then
        printf '%s\n' "$target" > "$fan/$out"
    fi
    [[ $(cat "$fan/$mode") == "$on" ]] || fail 'manual control readback failed'
    [[ $(read_rpm "$fan/$out") == "$target" ]] || fail 'fan target readback failed'
    # Report a change of policy, a settled step of 100 rpm or more and reclaimed control, not a few RPM of SMC jitter.
    if [[ $policy != "$logged_policy" || $manual != "$on" ]] \
        || (( target == want && (target - logged_target >= 100 || logged_target - target >= 100) )); then
        echo "fan target $target rpm ($policy; live SMC maximum $maximum)"
        logged_policy=$policy logged_target=$target
    fi
    sleep "$INTERVAL" & wait $!
done
