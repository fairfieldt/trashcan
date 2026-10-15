#!/usr/bin/env python3
"""How long a log line's write takes on this Mac: for SECONDS (or until FILE.stop exists), append a 200-byte line to
FILE every 5 ms (a write() of an ordinary file, as a process writes its stderr log), and print one JSON line: the
writes, the median, and every write over 20 ms with its time of day.
usage: wlat.py FILE SECONDS"""
import json, os, socket, sys, time

path, secs = sys.argv[1], float(sys.argv[2])
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
lat, slow = [], []
end = time.monotonic() + secs
line = b'x' * 199 + b'\n'
while time.monotonic() < end and not os.path.exists(path + '.stop'):
    t = time.monotonic()
    os.write(fd, line)
    d = time.monotonic() - t
    lat.append(d)
    if d > 0.02:
        slow.append([time.strftime('%H:%M:%S', time.localtime()) + f'.{int(time.time() * 1000) % 1000:03d}', round(d * 1e3, 1)])
    time.sleep(0.005)
os.close(fd)
for p in (path, path + '.stop'):
    if os.path.exists(p):
        os.remove(p)
lat.sort()
print(json.dumps({'host': socket.gethostname(), 'writes': len(lat), 'p50_ms': round(lat[len(lat) // 2] * 1e3, 3),
                  'max_ms': round(lat[-1] * 1e3, 1), 'slow': slow}), flush=True)
