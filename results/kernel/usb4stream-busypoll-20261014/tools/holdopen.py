#!/usr/bin/env python3
"""Hold stream devices open (so their DMA paths are up) for SECONDS, reading and dropping whatever arrives, then close
them. Prints one JSON line: per device whether it opened (or the errno), bytes and ends of file read, and read errors by
errno (a stock 7.3 end that read a stale CLOSE fails its reads with ENXIO).
usage: holdopen.py SECONDS DEV [DEV ...]"""
import errno, fcntl, json, os, sys, time

secs, devs = float(sys.argv[1]), sys.argv[2:]
res = {d: {'dev': d} for d in devs}
fds = {}
for d in devs:
    t = time.monotonic()
    try:
        fd = os.open(d, os.O_RDWR | os.O_NONBLOCK)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fds[d] = fd
        res[d].update(opened=True, open_ms=round((time.monotonic() - t) * 1e3, 1), bytes=0, eofs=0, errors={})
    except OSError as e:
        res[d].update(opened=False, error=errno.errorcode.get(e.errno, str(e.errno)))
end = time.monotonic() + secs
while time.monotonic() < end:
    for d, fd in fds.items():
        try:
            b = os.read(fd, 1 << 20)
            if b:
                res[d]['bytes'] += len(b)
            else:
                res[d]['eofs'] += 1
        except BlockingIOError:
            pass
        except OSError as e:
            k = errno.errorcode.get(e.errno, str(e.errno))
            res[d]['errors'][k] = res[d]['errors'].get(k, 0) + 1
    time.sleep(0.01)
for d, fd in fds.items():
    t = time.monotonic()
    os.close(fd)
    res[d]['close_ms'] = round((time.monotonic() - t) * 1e3, 1)
print(json.dumps({'step': 'holdopen', 'host': os.uname().nodename, 'secs': secs, 'devs': list(res.values())}), flush=True)
