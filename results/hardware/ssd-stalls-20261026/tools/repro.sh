#!/bin/bash
# repro.sh OUTDIR [WRITER]: cold model-load reproduction. Drops the page cache, then 4 parallel buffered readers of
# 5 GB each from the Flash-Next shards (20 GB, like a node's cold start) with an fsync writer (4 KiB every 100 ms)
# and an O_DIRECT 4 KiB random-read probe (a major fault) running from 3 s before to 3 s after.
# FSYNC=0 leaves the fsync writer out. WRITER=1 adds a background buffered writer (1 GB dd + sync), like logs/telemetry under load.
set -u
here=$(dirname "$(readlink -f "$0")"); out=$1; mkdir -p "$out"; cd "$out" || exit 1
 scratch=$HOME/trashcan-exp/ssd-stalls/scratch; mkdir -p "$scratch"
M=$HOME/models/qwen3.8-flash-next
files=("$M/G-iq4/G-iq4-00004-of-00006.gguf" "$M/G-iq4/G-iq4-00005-of-00006.gguf" "$M/G-iq4/G-iq4-00006-of-00006.gguf" "$M/Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf")
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; sleep 2
python3 "$here/sampler.py" sda 0.25 > disk.log & s=$!
if [ "${FSYNC:-1}" = 1 ]; then python3 "$here/probes.py" fsync fsync.log "$scratch/fsync-probe" & f=$!; else f=; : > fsync.log; echo "0 0" > fsync.log; fi
python3 "$here/probes.py" read read.log "$M/G-iq4/G-iq4-00002-of-00006.gguf" & r=$!
sleep 3
start=$(date +%s.%N)
pids=()
for i in 0 1 2 3; do dd if="${files[$i]}" of=/dev/null bs=4M count=1250 skip=$((i*500)) status=none & pids+=($!); done
if [ "${WRITER:-0}" = 1 ]; then (dd if=/dev/zero of="$scratch/bulk" bs=1M count=1024 status=none; sync -f "$scratch/bulk"; rm -f "$scratch/bulk") & pids+=($!); fi
wait "${pids[@]}"
end=$(date +%s.%N)
sleep 3; kill $s $f $r 2>/dev/null; wait 2>/dev/null
rm -f "$scratch/fsync-probe"
python3 - "$start" "$end" <<'PY'
import sys
s, e = float(sys.argv[1]), float(sys.argv[2])
def lat(fn):
    v = [float(l.split()[1]) for l in open(fn) if s - 3 <= float(l.split()[0]) <= e]
    v.sort(); n = len(v)
    if not n: return "off"
    return f"n={n} p50={v[n//2]:.1f} p99={v[int(n*.99)]:.1f} max={v[-1]:.1f} ms, >500ms: {sum(x>500 for x in v)}"
rows = [l.split() for l in open('disk.log')][1:]
rows = [r for r in rows if s <= float(r[0]) <= e]
zero = sum(1 for r in rows if float(r[1]) < 20) * 0.25
mb = sum(float(r[1]) for r in rows) * 0.25
fl = sum(float(r[5]) for r in rows) * 0.25
print(f"load: {e-s:.1f} s, {mb/(e-s):.0f} MB/s read; time with reads < 20 MB/s: {zero:.2f} s; flushes {fl:.0f}")
print("fsync probe:", lat('fsync.log'))
print("read probe :", lat('read.log'))
PY
