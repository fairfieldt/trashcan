#!/bin/bash
# Restart gpud serve on incite under lease $1 (with gpud.new if present). SIGTERM runs gpud's teardown; wait for it.
# GPUD_KMSG: every log line also reaches netconsole; GPUD_DEBUG: step markers too.
cd ~/trashcan-exp/gpud
sudo -n pkill -x gpud
for i in $(seq 1 50); do pgrep -x gpud > /dev/null || break; sleep 0.1; done
pgrep -x gpud > /dev/null && { echo "gpud still running"; exit 1; }
[ -f gpud.new ] && mv gpud.new gpud && chmod +x gpud
[ -f serve.log ] && mv serve.log serve.prev.log
nohup gpuq run "$1" -- sudo -n env GPUD_KMSG=1 GPUD_DEBUG=1 ./gpud serve 0000:02:00.0 > serve.log 2>&1 &
for i in $(seq 1 30); do grep -q "serving" serve.log 2>/dev/null && break; sleep 0.5; done
tail -1 serve.log
