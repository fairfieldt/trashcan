#!/bin/bash
# Reload thunderbolt.ko (built from ~/kernel/tb-throttle/src with the int_throttle_ns
# parameter) at each interval and run bench.sh pinned. Assumes performance governor
# and no_turbo=1 are already set.
# Usage: sweep-throttle.sh RUNPREFIX NS...
set -u
prefix=${1:?prefix}; shift
mod=~/kernel/tb-throttle/src/thunderbolt.ko
for t in "$@"; do
  sudo rmmod thunderbolt_net thunderbolt || exit 1
  sudo insmod $mod int_throttle_ns=$t || exit 1
  sudo modprobe thunderbolt_net
  sleep 5
  [ "$(cat /sys/module/thunderbolt/parameters/int_throttle_ns)" = "$t" ] || exit 1
  PIN=1 bash "$(dirname "$0")/bench.sh" "$prefix-$t" | tail -1
  sudo dmesg | grep -i -E "thunderbolt|tbnet" | tail -10 > "$prefix-$t/dmesg-tb.txt"
done
