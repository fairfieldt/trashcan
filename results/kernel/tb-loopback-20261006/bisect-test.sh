#!/bin/bash
# One bisect measurement: reset the controllers, load a module set, then count DMA path activation
# failures over N link bounces and M USB4STREAM setup/transfer cycles on the 0-3 <-> 1-3 cable.
# usage: bisect-test.sh MODDIR OUTDIR [N=20] [M=10]
set -u
moddir=$1 out=$2 n=${3:-20} m=${4:-10}
cd "$(dirname "$0")"; mkdir -p "$out"
fails() { sudo dmesg | grep -c "DMA tunnel activation failed\|failed to enable DMA paths\|timeout reading config space"; }
bash tb-reset.sh "$moddir" > "$out/reset.txt" 2>&1
{ uname -r; /usr/sbin/modinfo "$moddir/thunderbolt.ko" | grep -E "srcversion|vermagic"; cat "$moddir/VARIANT" 2>/dev/null; } > "$out/env.txt"
domains=$(ls -d /sys/bus/thunderbolt/devices/domain* 2>/dev/null | wc -l)
hosts=$(ls -d /sys/bus/thunderbolt/devices/[0-9]-[13] 2>/dev/null | wc -l)
f0=$(fails)
bash dma-repro.sh "$n" > "$out/bounce.txt" 2>&1
f1=$(fails)
cfg=/sys/kernel/config/thunderbolt/stream
ok=0
for i in $(seq "$m"); do
  sudo mkdir -p $cfg/0-3.1/data
  echo -1 | sudo tee $cfg/0-3.1/data/in_hopid $cfg/0-3.1/data/out_hopid >/dev/null
  # Configure the far end by hand: auto-configuration only copies the peer's HopIDs if its
  # property update has already arrived when the directory is created.
  sudo mkdir -p $cfg/1-3.1/data
  sudo cat $cfg/0-3.1/data/out_hopid | sudo tee $cfg/1-3.1/data/in_hopid >/dev/null
  sudo cat $cfg/0-3.1/data/in_hopid | sudo tee $cfg/1-3.1/data/out_hopid >/dev/null
  for d in 0-3.1/data 1-3.1/data; do echo "cycle $i $d $(for x in index in_hopid out_hopid; do echo -n "$x=$(sudo cat $cfg/$d/$x 2>/dev/null) "; done)"; done >> "$out/stream.txt"
  a=$(sudo cat $cfg/0-3.1/data/index 2>/dev/null) b=$(sudo cat $cfg/1-3.1/data/index 2>/dev/null)
  if sudo timeout -s KILL 15 ./tbstream-bench stream /dev/tbstream$a /dev/tbstream$b 65536 1 >> "$out/stream.txt" 2>&1 &&
     sudo timeout -s KILL 15 ./tbstream-bench pingpong /dev/tbstream$a /dev/tbstream$b 64 1000 >> "$out/stream.txt" 2>&1; then
    ok=$((ok + 1))
  else
    echo "cycle $i failed" >> "$out/stream.txt"
  fi
  sudo rmdir $cfg/1-3.1/data $cfg/0-3.1/data $cfg/1-3.1 $cfg/0-3.1 2>/dev/null
  sleep 1
done
f2=$(fails)
sudo dmesg | tail -200 > "$out/dmesg-tail.txt"
echo "domains=$domains hosts=$hosts bounce_fail_lines=$((f1 - f0))/$n stream_ok=$ok/$m stream_fail_lines=$((f2 - f1))" | tee "$out/summary.txt"
