#!/bin/bash
# Aggressive fan curve for incite (Mac Pro 6,1). One SMC fan cools both D700s and the CPU, so its speed follows
# the hottest of the two GPU edge sensors and the CPU package: minimum at or below T_LOW, linear to full speed at
# T_HIGH and above. A sensor that can't be read means full speed. Stopping the service hands the fan back to the SMC.
# Installed as /usr/local/sbin/fancurve.sh, run by fancurve.service.
set -u
T_LOW=${T_LOW:-40}     # °C
T_HIGH=${T_HIGH:-50}   # °C
INTERVAL=${INTERVAL:-2}
smc=/sys/devices/platform/applesmc.768
min=$(cat $smc/fan1_min)
max=$(cat $smc/fan1_max)

restore() {
    echo 0 > $smc/fan1_manual
    echo "fan back to SMC control"
    exit 0
}
trap restore TERM INT

hottest() {
    local f t m=0
    for f in /sys/class/drm/card[0-9]/device/hwmon/hwmon*/temp1_input \
             /sys/devices/platform/coretemp.0/hwmon/hwmon*/temp1_input; do
        t=$(cat "$f" 2>/dev/null) || return 1
        (( t > m )) && m=$t
    done
    echo $(( m / 1000 ))
}

echo "curve: ${min} rpm at <=${T_LOW} C, ${max} rpm at >=${T_HIGH} C"
last=
while :; do
    if t=$(hottest); then
        if (( t <= T_LOW )); then
            rpm=$min
        elif (( t >= T_HIGH )); then
            rpm=$max
        else
            rpm=$(( min + (max - min) * (t - T_LOW) / (T_HIGH - T_LOW) ))
        fi
    else
        t='?'
        rpm=$max
    fi
    # Re-assert manual mode if the SMC took it back.
    if [[ $(cat $smc/fan1_manual) != 1 ]]; then
        echo 1 > $smc/fan1_manual
        last=
    fi
    if [[ $rpm != "$last" ]]; then
        echo $rpm > $smc/fan1_output
        echo "hottest ${t} C -> ${rpm} rpm"
        last=$rpm
    fi
    sleep "$INTERVAL" & wait $!
done
