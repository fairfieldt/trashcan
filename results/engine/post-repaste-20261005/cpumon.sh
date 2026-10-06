#!/bin/bash
# Samples mean/max CPU MHz and package temp (C) once per second. usage: cpumon.sh OUTFILE
h=$(grep -l coretemp /sys/class/hwmon/hwmon*/name | head -1 | xargs dirname)
echo "t mhz_mean mhz_max pkg_temp" > "$1"
while :; do
  echo "$(date +%s) $(awk -F: '/MHz/{s+=$2;n++;if($2>m)m=$2}END{printf "%.0f %.0f", s/n, m}' /proc/cpuinfo) $(( $(cat $h/temp1_input)/1000 ))" >> "$1"
  sleep 1
done
