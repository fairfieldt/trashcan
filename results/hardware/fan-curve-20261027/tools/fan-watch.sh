#!/bin/bash
# Log fan rpm/target, both GPU edge temperatures, CPU package and the DPM level on every Mac, every STEP seconds
# for MIN minutes. usage: fan-watch.sh OUT MIN [STEP] [MACHINES]
out=$1 min=$2 step=${3:-15} machines=${4:-"incite invite indite enlight inflight alight"}
read -r -d '' R <<'X'
H=$(echo /sys/devices/platform/applesmc.768/hwmon/hwmon*)
g=""; for c in /sys/class/drm/card0 /sys/class/drm/card1; do g+="$(( $(cat $c/device/hwmon/hwmon*/temp1_input)/1000 )) "; done
for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name)" = coretemp ] && cpu=$(( $(cat $h/temp1_input)/1000 )); done
echo "fan $(cat $H/fan1_input)/$(cat $H/fan1_target) gpu ${g% } cpu $cpu dpm $(cat /sys/class/drm/card0/device/power_dpm_force_performance_level)"
X
end=$(( $(date +%s) + min * 60 ))
while (( $(date +%s) < end )); do
    for m in $machines; do echo "$(date +%T) $m $(ssh -o ConnectTimeout=5 $m "$R" 2>&1 | tail -1)" & done
    wait
    sleep $step
done >> $out
