#!/usr/bin/env python3
"""A CLOSE sent while the other end is open and not reading, then a new session at both ends: does the new session
see the old CLOSE? (7.3 without 0006: every later read end of file, writes ENXIO; 7.4 mainline: writes EPIPE and
EPOLLHUP once a CLOSE lands in the ring.) One end of the cable per call; the Mac sequences the steps.

usage:
  closestale.py hold DEV SECS            open, wait SECS without reading, close (prints rx bytes left unread: none read)
  closestale.py openclose DEV            open and close at once (sends a CLOSE)
  closestale.py new DEV MODE WAIT        open (MODE nb|block), poll() at once, wait for the peer's frame up to WAIT s,
                                         read what is there, write one 112-byte frame, poll() again, close
  closestale.py peer DEV MODE DELAY      open, sleep DELAY, write one 112-byte frame, read the reply up to 1 s, close
"""
import errno, fcntl, json, os, select, sys, time

FRAME = bytes(range(112))


def out(**kw):
    kw['host'] = os.uname().nodename
    print(json.dumps(kw), flush=True)


def open_dev(dev, mode):
    flags = os.O_RDWR | (os.O_NONBLOCK if mode == 'nb' else 0)
    fd = os.open(dev, flags)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def pollmask(fd, ms=0):
    p = select.poll()
    p.register(fd, select.POLLIN | select.POLLOUT | select.POLLHUP | select.POLLERR)
    r = p.poll(ms)
    m = r[0][1] if r else 0
    return [n for n, b in (('IN', select.POLLIN), ('OUT', select.POLLOUT), ('HUP', select.POLLHUP), ('ERR', select.POLLERR)) if m & b]


def err(e):
    return errno.errorcode.get(e.errno, str(e.errno))


def read_some(fd, mode, wait):
    """Bytes and ends of file until the first data or WAIT s (blocking reads are bounded by poll)."""
    got = eofs = 0
    events = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait and got == 0:
        if mode == 'block':
            if not pollmask(fd, 50):
                continue
        try:
            b = os.read(fd, 1 << 16)
        except BlockingIOError:
            time.sleep(0.001)
            continue
        except OSError as e:
            events.append('read ' + err(e))
            break
        if b:
            got += len(b)
            events.append('data %d at %.2f ms' % (len(b), (time.monotonic() - t0) * 1e3))
        else:
            eofs += 1
            if eofs <= 3:
                events.append('eof')
            if eofs >= 1000:
                break
            if mode == 'block':
                time.sleep(0.001)
    return got, eofs, events


def main():
    cmd, dev = sys.argv[1], sys.argv[2]
    if cmd == 'hold':
        fd = open_dev(dev, 'nb')
        time.sleep(float(sys.argv[3]))
        before = pollmask(fd)
        os.close(fd)
        out(step='hold', poll_before_close=before)
    elif cmd == 'openclose':
        fd = open_dev(dev, 'nb')
        os.close(fd)
        out(step='openclose')
    elif cmd == 'new':
        mode, wait = sys.argv[3], float(sys.argv[4])
        fd = open_dev(dev, mode)
        p0 = pollmask(fd)
        got, eofs, events = read_some(fd, mode, wait)
        t_data = time.time()
        try:
            os.write(fd, FRAME)
            wrote = 'ok'
        except OSError as e:
            wrote = err(e)
        p1 = pollmask(fd)
        got2, eofs2, ev2 = read_some(fd, 'nb', 0.2)
        os.close(fd)
        out(step='new', mode=mode, data_at=t_data, poll_open=p0, bytes=got, eofs=eofs, events=events, write=wrote, poll_after=p1,
            bytes_after=got2, eofs_after=eofs2)
    elif cmd == 'peer':
        mode, delay = sys.argv[3], float(sys.argv[4])
        fd = open_dev(dev, mode)
        time.sleep(delay)
        tw = time.time()
        try:
            os.write(fd, FRAME)
            wrote = 'ok'
        except OSError as e:
            wrote = err(e)
        got, eofs, events = read_some(fd, 'nb', 1.0)
        os.close(fd)
        out(step='peer', mode=mode, write=wrote, wrote_at=tw, bytes=got, eofs=eofs, events=events)
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
