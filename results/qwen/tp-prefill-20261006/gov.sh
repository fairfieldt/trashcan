#!/bin/bash
# usage: gov.sh performance|schedutil
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo "$1" | sudo -n tee "$c" >/dev/null; done
sort /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | uniq -c
