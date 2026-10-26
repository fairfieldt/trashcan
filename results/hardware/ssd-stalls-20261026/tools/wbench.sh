#!/bin/bash
# wbench.sh: write costs on an idle disk. Sequential 4 GiB write + fdatasync, 200 x (4 KiB append + fsync),
# 3000 files of 16 KiB then one sync (dpkg/untar-like), in ~/trashcan-exp/ssd-stalls/scratch; removes its files.
set -eu
d=$HOME/trashcan-exp/ssd-stalls/scratch; mkdir -p "$d"; cd "$d"
t0=$(date +%s.%N); dd if=/dev/zero of=seq bs=4M count=1024 conv=fdatasync status=none; t1=$(date +%s.%N)
echo "seq 4 GiB write+fdatasync: $(echo "4096/($t1-$t0)" | bc -l | cut -c1-6) MB/s"; rm -f seq; sync
python3 - <<'PY'
import os, time
fd = os.open('fs', os.O_WRONLY | os.O_CREAT | os.O_APPEND); v = []
for i in range(200):
    t = time.time(); os.write(fd, b'x' * 4096); os.fsync(fd); v.append((time.time() - t) * 1000)
os.close(fd); os.unlink('fs'); v.sort()
print(f"4 KiB append+fsync x200: p50 {v[100]:.1f} p99 {v[198]:.1f} max {v[-1]:.1f} ms")
os.makedirs('many', exist_ok=True); b = b'y' * 16384; t = time.time()
for i in range(3000):
    with open(f'many/{i}', 'wb') as f: f.write(b)
os.sync(); print(f"3000 x 16 KiB files + sync: {time.time() - t:.2f} s")
PY
rm -rf many; sync
