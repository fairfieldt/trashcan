#!/bin/bash
# One USB4STREAM per cable on the three-cable triangle (0-3/1-3, 0-1/2-3, 1-1/2-1): each alone, then all three at
# once in one direction, then all three both ways. Needs thunderbolt_stream loaded.
# usage: stream-tri.sh OUTDIR
set -u
out=$1; cd "$(dirname "$0")"; mkdir -p "$out"
cfg=/sys/kernel/config/thunderbolt/stream
pairs="0-3:1-3 0-1:2-3 1-1:2-1"
for p in $pairs; do
  a=${p%%:*}.1 b=${p##*:}.1
  sudo mkdir -p $cfg/$a/data $cfg/$b/data
  [ "$(sudo cat $cfg/$a/data/in_hopid)" -gt 0 ] 2>/dev/null || echo -1 | sudo tee $cfg/$a/data/in_hopid $cfg/$a/data/out_hopid >/dev/null
  sudo cat $cfg/$a/data/out_hopid | sudo tee $cfg/$b/data/in_hopid >/dev/null 2>&1
  sudo cat $cfg/$a/data/in_hopid | sudo tee $cfg/$b/data/out_hopid >/dev/null 2>&1
done
dev() { echo /dev/tbstream$(sudo cat $cfg/$1.1/data/index); }
{
  for p in $pairs; do echo "== $p alone"; sudo ./tbstream-bench stream $(dev ${p%%:*}) $(dev ${p##*:}) 1048576 5; done
  echo "== all three, one direction"
  for p in $pairs; do sudo ./tbstream-bench stream $(dev ${p%%:*}) $(dev ${p##*:}) 1048576 10 & done; wait
  echo "== all three, both directions (six streams)"
  for p in $pairs; do
    sudo ./tbstream-bench stream $(dev ${p%%:*}) $(dev ${p##*:}) 1048576 10 &
  done
  for p in $pairs; do
    a=${p%%:*} b=${p##*:}
    sudo mkdir -p $cfg/$a.1/back $cfg/$b.1/back
    [ "$(sudo cat $cfg/$a.1/back/in_hopid)" -gt 0 ] 2>/dev/null || echo -1 | sudo tee $cfg/$a.1/back/in_hopid $cfg/$a.1/back/out_hopid >/dev/null
    sudo cat $cfg/$a.1/back/out_hopid | sudo tee $cfg/$b.1/back/in_hopid >/dev/null 2>&1
    sudo cat $cfg/$a.1/back/in_hopid | sudo tee $cfg/$b.1/back/out_hopid >/dev/null 2>&1
    sudo ./tbstream-bench stream /dev/tbstream$(sudo cat $cfg/$b.1/back/index) /dev/tbstream$(sudo cat $cfg/$a.1/back/index) 1048576 8 &
  done; wait
} 2>&1 | tee "$out/tri.txt"
