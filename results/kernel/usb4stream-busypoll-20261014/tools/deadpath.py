#!/usr/bin/env python3
"""Does writing toward an end whose device is closed leave that direction dead once it opens? (one end of a cable;
the other end runs the other role with the same START, a unix time; the Macs' clocks agree to well under 10 ms).

usage:
  deadpath.py early DEV START K OPEN TEST END RUN   at START open DEV and write K data frames (the other end is still
                                                    closed), at START+TEST write 20 test frames, read until START+END
  deadpath.py late DEV START K OPEN TEST END RUN    at START+OPEN open DEV, at START+TEST write 20 test frames, read
                                                    until START+END
Each prints what it wrote (and EAGAINs) and what it read of the other side: burst frames, test frames, ends of file.
112-byte frames: magic, kind (1 burst, 2 test), number, run, pattern, CRC32. A receiving NHI holds its last frames
until more follow, so 17 of 20 test frames read is a working direction.
"""
import fcntl, json, os, struct, sys, time, zlib

MAGIC = 0x48544150  # "PATH"
SIZE = 112
TESTS = 20


def frame(kind, i, run):
    body = struct.pack('<IIII', MAGIC, kind, i, run) + bytes((j * 13 + 7) & 0xFF for j in range(SIZE - 20))
    return body + struct.pack('<I', zlib.crc32(body))


def until(t):
    while True:
        d = t - time.time()
        if d <= 0:
            return
        time.sleep(min(d, 0.01))


def main():
    role, dev = sys.argv[1], sys.argv[2]
    start, k, t_open, t_test, t_end, run = float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), \
        float(sys.argv[7]), int(sys.argv[8])
    out = {'role': role, 'dev': dev, 'k': k, 'open': t_open, 'run': run, 'host': os.uname().nodename}
    until(start + (0 if role == 'early' else t_open))
    fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out['opened_late_ms'] = round((time.time() - start - (0 if role == 'early' else t_open)) * 1e3, 1)
    buf = bytearray()
    got = {1: set(), 2: set()}
    other = eofs = 0

    def write_all(kind, n, limit):
        sent = eagain = 0
        t0 = time.time()
        while sent < n and time.time() - t0 < limit:
            try:
                os.write(fd, frame(kind, sent, run))
                sent += 1
            except BlockingIOError:
                eagain += 1
                time.sleep(0.0005)
        return sent, eagain

    def read_until(t):
        nonlocal buf, other, eofs
        m = struct.pack('<I', MAGIC)
        while time.time() < t:
            try:
                b = os.read(fd, 1 << 20)
            except BlockingIOError:
                time.sleep(0.0005)
                continue
            if not b:
                eofs += 1
                time.sleep(0.001)
                continue
            buf += b
            while True:
                j = buf.find(m)
                if j < 0 or len(buf) < j + SIZE:
                    break
                f = bytes(buf[j:j + SIZE])
                if zlib.crc32(f[:-4]) == struct.unpack('<I', f[-4:])[0]:
                    _, kind, i, r = struct.unpack('<IIII', f[:16])
                    if r == run and kind in got:
                        got[kind].add(i)
                    else:
                        other += 1
                    del buf[:j + SIZE]
                else:
                    del buf[:j + 1]

    if role == 'early' and k >= 0:
        out['burst_written'], out['burst_eagain'] = write_all(1, k, 0.2)
    elif role == 'early':
        # K < 0: a frame every millisecond until the test, so frames arrive while the late end opens.
        sent = eagain = 0
        while time.time() < start + t_test - 0.1:
            try:
                os.write(fd, frame(1, sent, run))
                sent += 1
            except BlockingIOError:
                eagain += 1
            read_until(time.time() + 0.001)
        out['burst_written'], out['burst_eagain'] = sent, eagain
    read_until(start + t_test)
    out['test_written'], out['test_eagain'] = write_all(2, TESTS, 0.2)
    read_until(start + t_end)
    t = time.time()
    os.close(fd)
    out.update(burst_read=len(got[1]), test_read=len(got[2]), other=other, eofs=eofs,
               close_ms=round((time.time() - t) * 1e3, 1))
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
