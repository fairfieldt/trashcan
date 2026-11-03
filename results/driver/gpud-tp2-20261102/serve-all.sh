#!/bin/bash
# Start one gpud serve per GPU given (bus numbers, e.g. 02 06) after stopping any running gpud (SIGTERM runs the
# teardown). Each serves /run/gpud/ADDRESS.sock and logs to serve-BUS.log. Run under the incite gpuq lease.
cd ~/trashcan-exp/gpud
sudo -n pkill -x gpud
for i in $(seq 1 50); do pgrep -x gpud > /dev/null || break; sleep 0.1; done
pgrep -x gpud > /dev/null && { echo "gpud still running"; exit 1; }
for b in "$@"; do
  [ -f serve-$b.log ] && mv serve-$b.log serve-$b.prev.log
  nohup sudo -n env GPUD_KMSG=1 GPUD_DEBUG=1 ./gpud serve 0000:$b:00.0 > serve-$b.log 2>&1 &
done
for b in "$@"; do
  for i in $(seq 1 40); do grep -q "serving" serve-$b.log 2>/dev/null && break; sleep 0.5; done
  echo "$b: $(tail -1 serve-$b.log)"
done
