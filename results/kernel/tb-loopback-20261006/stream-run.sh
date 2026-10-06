#!/bin/bash
# USB4STREAM benchmark between /dev/tbstream0 (0-3.1/data) and /dev/tbstream1 (1-3.1/data).
# usage: stream-run.sh OUTDIR THROTTLE_NS
set -u
out=$1 thr=$2
cd "$(dirname "$0")"; mkdir -p "$out"
cfg=/sys/kernel/config/thunderbolt/stream
for d in 0-3.1/data 1-3.1/data; do echo "$thr" | sudo tee $cfg/$d/throttling >/dev/null; done
{ for d in 0-3.1/data 1-3.1/data; do echo "$d $(for a in index in_hopid out_hopid ring_size throttling; do echo -n "$a=$(sudo cat $cfg/$d/$a) "; done)"; done
  echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"; uname -r; } > "$out/env.txt"
B="sudo ./tbstream-bench"
for sz in 64 1024 4096; do $B pingpong /dev/tbstream0 /dev/tbstream1 $sz 20000; done > "$out/pingpong.txt" 2>&1
for ch in 4096 65536 1048576; do $B stream /dev/tbstream0 /dev/tbstream1 $ch 10; done > "$out/stream-fwd.txt" 2>&1
$B stream /dev/tbstream1 /dev/tbstream0 1048576 10 > "$out/stream-rev.txt" 2>&1
cat "$out"/pingpong.txt "$out"/stream-*.txt
