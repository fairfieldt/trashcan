#!/bin/bash
# Two concurrent USB4STREAMs on the 0-3 <-> 1-3 cable: same direction, then opposite directions.
# usage: stream-pair.sh OUTDIR
set -u
out=$1; cd "$(dirname "$0")"; mkdir -p "$out"
cfg=/sys/kernel/config/thunderbolt/stream
for s in data data2; do
  sudo mkdir -p $cfg/0-3.1/$s $cfg/1-3.1/$s
  [ "$(sudo cat $cfg/0-3.1/$s/in_hopid)" -gt 0 ] 2>/dev/null || echo -1 | sudo tee $cfg/0-3.1/$s/in_hopid $cfg/0-3.1/$s/out_hopid >/dev/null
  sudo cat $cfg/0-3.1/$s/out_hopid | sudo tee $cfg/1-3.1/$s/in_hopid >/dev/null
  sudo cat $cfg/0-3.1/$s/in_hopid | sudo tee $cfg/1-3.1/$s/out_hopid >/dev/null
done
for d in 0-3.1/data 0-3.1/data2 1-3.1/data 1-3.1/data2; do echo "$d $(for x in index in_hopid out_hopid; do echo -n "$x=$(sudo cat $cfg/$d/$x) "; done)"; done | tee "$out/config.txt"
dev() { echo /dev/tbstream$(sudo cat $cfg/$1/index); }
A1=$(dev 0-3.1/data) B1=$(dev 1-3.1/data) A2=$(dev 0-3.1/data2) B2=$(dev 1-3.1/data2)
{ echo "== single"; sudo ./tbstream-bench stream $A1 $B1 1048576 10
  echo "== two, same direction"
  sudo ./tbstream-bench stream $A1 $B1 1048576 10 & sudo ./tbstream-bench stream $A2 $B2 1048576 10; wait
  echo "== two, opposite directions"
  sudo ./tbstream-bench stream $A1 $B1 1048576 10 & sudo ./tbstream-bench stream $B2 $A2 1048576 10; wait
} 2>&1 | tee "$out/pair.txt"
