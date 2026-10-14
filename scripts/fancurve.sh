#!/bin/bash
# Keep the Mac Pro 6,1 main fan at FAN_RPM (default 2350) or the live SMC maximum, whichever is higher,
# at every temperature. The SMC's maximum (fan1_max) is 1900 at boot and drifts between 1900 and ~2380
# afterwards; the SMC accepts a manual target above it. FAN_RPM=0 follows the SMC maximum alone.
# fan1_max can change after boot, so never cache a successful reading indefinitely.
# Stop/failure returns control to the SMC. Installed as /usr/local/sbin/fancurve.sh.
set -euo pipefail
INTERVAL=${INTERVAL:-2}
FAN_RPM=${FAN_RPM:-2350}
smc=${SMC_PATH:-/sys/devices/platform/applesmc.768}
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

[[ $INTERVAL =~ ^([1-9]|10)$ ]] || fail 'INTERVAL must be 1..10 seconds'
# 2400 rpm caps typos; the SMC's own maximum has reached 2382.
[[ $FAN_RPM =~ ^(0|[1-9][0-9]{0,3})$ ]] && (( FAN_RPM <= 2400 )) || fail 'FAN_RPM must be 0..2400'
minimum=$(read_rpm "$fan/fan1_min") || fail 'cannot read a valid SMC minimum'
(( minimum > 0 )) || fail 'invalid SMC minimum'
last_max= failures=0 logged_target=
echo "fan policy: at least $FAN_RPM rpm, or the live SMC maximum if higher, at all temperatures; refresh every ${INTERVAL}s"
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
    target=$(( maximum > FAN_RPM ? maximum : FAN_RPM ))
    manual=$(cat "$fan/$mode") || fail 'cannot read fan control mode'
    [[ $manual == "$off" || $manual == "$on" ]] || fail 'invalid fan control mode'
    output=$(read_rpm "$fan/$out") || fail 'cannot read fan target'
    if [[ $manual != "$on" ]]; then printf '%s\n' "$on" > "$fan/$mode"; fi
    if [[ $manual != "$on" || $output != "$target" ]]; then
        printf '%s\n' "$target" > "$fan/$out"
    fi
    [[ $(cat "$fan/$mode") == "$on" ]] || fail 'manual control readback failed'
    [[ $(read_rpm "$fan/$out") == "$target" ]] || fail 'fan target readback failed'
    # Report policy changes and reclaimed control, not a few RPM of SMC jitter.
    if [[ -z $logged_target || $manual != "$on" ]] || (( target - logged_target >= 50 || logged_target - target >= 50 )); then
        echo "fan target $target rpm (live SMC maximum $maximum)"
        logged_target=$target
    fi
    sleep "$INTERVAL" & wait $!
done
