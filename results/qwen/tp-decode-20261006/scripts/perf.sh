#!/bin/bash
# usage: perf.sh performance|schedutil
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo $1 | sudo tee $c >/dev/null; done
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
