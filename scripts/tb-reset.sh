#!/bin/bash
# Reset the Thunderbolt controllers of several Macs together, so every link between them renegotiates at once (run from
# this Mac). On every Mac: remove the stream groups, unload the Thunderbolt modules and reset the three NHIs; once all
# are done, load the modules on all of them within a second; then list each Mac's links. Resetting one Mac after
# another leaves its neighbours' ends stale or on one lane; together they came back on two lanes, five links of six
# after one round and the last after a second (results/qwen/tp8-mesh-20261009). A link still missing after two rounds
# needs its cable replugged. Streams must be created again afterwards (scripts/tbstream-up.sh).
# usage: tb-reset.sh HOST...      (e.g. tb-reset.sh incite invite enlight indite)
set -uo pipefail
[ "$#" -ge 1 ] || { echo 'usage: tb-reset.sh HOST...' >&2; exit 1; }
unload='sudo rmdir /sys/kernel/config/thunderbolt/stream/*/* /sys/kernel/config/thunderbolt/stream/* 2>/dev/null
sudo rmmod thunderbolt_ibverbs thunderbolt_stream thunderbolt_dma_test thunderbolt_net 2>/dev/null
sudo rmmod thunderbolt || { echo "$(hostname): rmmod thunderbolt failed" >&2; exit 1; }
for nhi in 16:00.0 5d:00.0 a4:00.0; do sudo sh -c "echo 1 > /sys/bus/pci/devices/0000:$nhi/reset"; done'
load='sudo modprobe thunderbolt_net && sudo modprobe thunderbolt_stream'
each() {
  local failed=0 pids=() h
  for h in "$@"; do ssh "$h" "$step" & pids+=($!); done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return $failed
}
step=$unload each "$@" || { echo 'unloading failed on a Mac; modules left unloaded there' >&2; exit 1; }
sleep 3
step=$load each "$@" || { echo 'loading the modules failed on a Mac' >&2; exit 1; }
sleep 20
for h in "$@"; do
  echo "== $h"
  ssh "$h" /usr/local/sbin/tblink links | tail -n +2
done
