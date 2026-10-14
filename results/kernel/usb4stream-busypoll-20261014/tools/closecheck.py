#!/usr/bin/env python3
"""A CLOSE that waits at a closed end, and what the device does after reading it (one end of a cable; the other end
runs the matching step, never at the same time).

usage:
  closecheck.py plant DEV N        open and close DEV N times, the other end closed: each close sends a CLOSE, which
                                   waits there for its next open
  closecheck.py check DEV SECONDS  open DEV and read for SECONDS (bytes, ends of file), write one 112-byte frame (ok
                                   or the error), read 0.2 s more (ends of file after the write), close

On 7.2.9 a CLOSE read gives one end of file and the device goes on working. On 7.3-rc7 every later read returns end
of file and every write fails with ENXIO until the device is closed (patches-7.3/0006 restores 7.2's behaviour).
"""
import errno, fcntl, json, os, sys, time

STICKY = 10000  # ends of file in a row that count as "every read"


def open_dev(dev):
    fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def read_for(fd, secs):
    """Bytes and ends of file read in `secs` (ends of file stop counting at STICKY)."""
    n = eofs = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        try:
            b = os.read(fd, 1 << 20)
        except BlockingIOError:
            time.sleep(0.001)
            continue
        if b:
            n += len(b)
        else:
            eofs += 1
            if eofs >= STICKY:
                break
    return n, eofs


def out(**kw):
    kw['host'] = os.uname().nodename
    print(json.dumps(kw), flush=True)


def main():
    cmd, dev = sys.argv[1], sys.argv[2]
    if cmd == 'plant':
        times = []
        for _ in range(int(sys.argv[3])):
            fd = open_dev(dev)
            t = time.monotonic()
            os.close(fd)
            times.append(round((time.monotonic() - t) * 1e3, 1))
        out(step='plant', dev=dev, close_ms=times)
    elif cmd == 'check':
        fd = open_dev(dev)
        got, eofs = read_for(fd, float(sys.argv[3]))
        try:
            os.write(fd, bytes(112))
            wrote = 'ok'
        except OSError as e:
            wrote = errno.errorcode.get(e.errno, str(e.errno))
        got_after, eofs_after = read_for(fd, 0.2)
        t = time.monotonic()
        os.close(fd)
        out(step='check', dev=dev, bytes=got, eofs=eofs, write=wrote, bytes_after=got_after, eofs_after=eofs_after,
            sticky=eofs_after >= STICKY, close_ms=round((time.monotonic() - t) * 1e3, 1))
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
