#!/bin/bash
# Wait until every D700 of the given Macs reads at most LIMIT C (the sensor scripts/clockmon.sh logs), so that each run
# starts from the same temperature: invite's first card throttles from about 90 C, which MTP runs reach (README).
# usage: cool.sh LIMIT HOST...
limit=$1; shift
start=$(date +%s)
while :; do
  hot=0 temps=
  for h in "$@"; do
    t=$(ssh "$h" 'cat /sys/class/drm/card[01]/device/hwmon/hwmon*/temp1_input' | awk '{printf "%d ", $1 / 1000}')
    temps+="$h $t"
    for x in $t; do [ "$x" -gt "$limit" ] && hot=1; done
  done
  [ $hot = 0 ] && break
  sleep 10
done
echo "cool after $(( $(date +%s) - start )) s: $temps"
