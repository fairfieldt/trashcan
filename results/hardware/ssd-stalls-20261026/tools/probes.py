#!/usr/bin/env python3
"""probes.py MODE OUT ARG: 'fsync' appends 4 KiB + fsync to ARG every 100 ms; 'read' does an O_DIRECT 4 KiB pread
at a random offset of file ARG every 50 ms. Prints 't latency_ms' per op until killed."""
import os, sys, time, random, mmap
mode, out, arg = sys.argv[1:4]
o = open(out, 'w', buffering=1)
if mode == 'fsync':
    fd = os.open(arg, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644); buf = b'x' * 4096; gap = 0.1
else:
    fd = os.open(arg, os.O_RDONLY | os.O_DIRECT); size = os.fstat(fd).st_size
    buf = mmap.mmap(-1, 4096); gap = 0.05
while True:
    t = time.time()
    if mode == 'fsync':
        os.write(fd, buf); os.fsync(fd)
    else:
        os.preadv(fd, [buf], random.randrange(size // 4096) * 4096)
    d = time.time() - t
    o.write(f'{t:.3f} {d*1000:.1f}\n')
    time.sleep(max(0, gap - d))
