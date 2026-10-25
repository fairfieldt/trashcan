#!/bin/bash
# Reset the Thunderbolt controllers of several Macs together, so every link between them renegotiates at once (run from
# your workstation, with SSH and passwordless sudo on every Mac). On every Mac: stop tblink, remove the stream groups, unload the Thunderbolt modules and reset the three
# NHIs; once all are done, load the modules on all of them within a second and start tblink again, which addresses the
# links and creates their streams; then list each Mac's links. Resetting one Mac after another leaves its neighbours'
# ends stale or on one lane; together they came back on two lanes, five links of six after one round and the last
# after a second (four Macs, six cables). A link still missing after two rounds needs its cable replugged.
# A process with a stream device open keeps thunderbolt_stream loaded, and the reset then fails on that Mac.
# usage: tb-reset.sh HOST...      (e.g. tb-reset.sh mac0 mac1 mac2 mac3)
set -uo pipefail
[ "$#" -ge 1 ] || { echo 'usage: tb-reset.sh HOST...' >&2; exit 1; }
unload='sudo systemctl stop tblink.service 2>/dev/null
sudo rmdir /sys/kernel/config/thunderbolt/stream/*/* /sys/kernel/config/thunderbolt/stream/* 2>/dev/null
sudo rmmod thunderbolt_ibverbs thunderbolt_stream thunderbolt_dma_test thunderbolt_net 2>/dev/null
sudo rmmod thunderbolt || { echo "$(hostname): rmmod thunderbolt failed" >&2; exit 1; }
for nhi in 16:00.0 5d:00.0 a4:00.0; do sudo sh -c "echo 1 > /sys/bus/pci/devices/0000:$nhi/reset"; done'
load='sudo modprobe thunderbolt_net && sudo modprobe thunderbolt_stream && sudo systemctl start tblink.service'
each() {
  local failed=0 pids=() h
  for h in "$@"; do ssh "$h" "$step" & pids+=($!); done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return $failed
}
if ! step=$unload each "$@"; then
  echo 'unloading failed on a Mac; loading the modules and starting tblink again everywhere' >&2
  step=$load each "$@"
  exit 1
fi
sleep 3
step=$load each "$@" || { echo 'loading the modules failed on a Mac' >&2; exit 1; }
sleep 25
for h in "$@"; do
  echo "== $h"
  ssh "$h" /usr/local/sbin/tblink links | tail -n +2
done
