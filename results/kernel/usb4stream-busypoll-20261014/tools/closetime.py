#!/usr/bin/env python3
"""How long closing a stream device takes, and how long opening it again waits meanwhile (one end of a cable).

usage:
  closetime.py hold DEV SECONDS   open DEV, read for SECONDS, then close it on a thread and open it again 20 ms later:
                                  the close's duration and the reopen's wait (the driver's lock is held by the close)
  closetime.py die DEV SECONDS    open DEV, write 4 frames, and after SECONDS die with SIGKILL (the kernel closes it)
"""
import json, os, sys, threading, time


def main():
    cmd, dev, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
    fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
    if cmd == 'die':
        for _ in range(4):
            os.write(fd, bytes(112))
        time.sleep(secs)
        os.kill(os.getpid(), 9)
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
            time.sleep(0.001)
    res = {}

    def closer():
        t = time.monotonic()
        os.close(fd)
        res['close_ms'] = round((time.monotonic() - t) * 1e3, 1)

    th = threading.Thread(target=closer)
    th.start()
    time.sleep(0.02)
    t = time.monotonic()
    fd2 = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
    res['reopen_wait_ms'] = round((time.monotonic() - t) * 1e3, 1)
    th.join()
    t = time.monotonic()
    os.close(fd2)
    res['close2_ms'] = round((time.monotonic() - t) * 1e3, 1)
    print(json.dumps(dict(res, step=cmd, dev=dev, bytes=n, eofs=eofs, host=os.uname().nodename)), flush=True)


if __name__ == '__main__':
    main()
