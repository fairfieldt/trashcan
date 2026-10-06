#!/bin/bash
# One USB4STREAM setup + open + small transfer on 0-3 <-> 1-3 with thunderbolt dynamic debug on.
# usage: stream-once.sh MODDIR OUTDIR
set -u
moddir=$1 out=$2
cd "$(dirname "$0")"; mkdir -p "$out"
bash tb-reset.sh "$moddir" > "$out/reset.txt" 2>&1
dd=/sys/kernel/debug/dynamic_debug/control
echo "module thunderbolt +pt" | sudo tee $dd >/dev/null
echo "module thunderbolt_stream +pt" | sudo tee $dd >/dev/null
mark=$(awk '{print $1}' /proc/uptime)
cfg=/sys/kernel/config/thunderbolt/stream
sudo mkdir -p $cfg/0-3.1/data
echo -1 | sudo tee $cfg/0-3.1/data/in_hopid $cfg/0-3.1/data/out_hopid >/dev/null
for i in $(seq 20); do sleep 0.5; [ -d $cfg/1-3.1 ] || sudo mkdir -p $cfg/1-3.1/data 2>/dev/null; b=$(sudo cat $cfg/1-3.1/data/in_hopid 2>/dev/null); [ -n "$b" ] && [ "$b" != 0 ] && [ "$b" != -1 ] && break; done
for d in 0-3.1/data 1-3.1/data; do echo "$d $(for a in index in_hopid out_hopid; do echo -n "$a=$(sudo cat $cfg/$d/$a) "; done)"; done | tee "$out/config.txt"
a=$(sudo cat $cfg/0-3.1/data/index) b=$(sudo cat $cfg/1-3.1/data/index)
sudo timeout -s KILL 15 ./tbstream-bench pingpong /dev/tbstream$a /dev/tbstream$b 64 100 2>&1 | tee "$out/pingpong.txt"
echo "module thunderbolt -p" | sudo tee $dd >/dev/null
echo "module thunderbolt_stream -p" | sudo tee $dd >/dev/null
sudo dmesg | awk -F'[][]' -v m="$mark" '$2+0 >= m' > "$out/dmesg.txt"
wc -l < "$out/dmesg.txt"
