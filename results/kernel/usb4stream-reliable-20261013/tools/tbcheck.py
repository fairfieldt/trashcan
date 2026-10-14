#!/usr/bin/env python3
"""USB4STREAM driver behaviour checks on one end of a cable (the other end runs the matching step).

Frames are 4096 bytes: magic, sequence number, then pattern, with a CRC32 in the last 4 bytes, written one per write.
usage:
  tbcheck.py opentime DEV               open/close durations with the other end closed (5 times)
  tbcheck.py writer DEV N HOLD          open, write N frames (non-blocking; count EAGAIN), hold HOLD s, close (timed)
  tbcheck.py reader DEV DELAY SECONDS   sleep DELAY, open, read for SECONDS: frames intact/damaged, EOFs, sequence range
  tbcheck.py lazy DEV DELAY SECONDS     open, sleep DELAY without reading (the ring fills), then read as `reader`
  tbcheck.py blast DEV N                open, write N frames as fast as the Tx ring takes them, close (timed)
  tbcheck.py pp DEV client|server N SIZE   ping-pong of single SIZE-byte frames, both ends spinning on non-blocking
                                        reads; a side that waits 20 ms after its last write writes a filler frame, so
                                        a frame the receiver does not see by itself shows up as fillers and time
  tbcheck.py drain DEV SECONDS          open, read whatever the device holds for SECONDS (bytes, ends of file), close
  tbcheck.py pos DEV client|server ROUNDS RUN   the hold against the receiving ring's position: the client writes
                                        0-3 fillers (round by round), then a lone frame, then one filler every 20 ms
                                        until the server answers (each answer followed by three fillers, so answers
                                        never wait); per round, the client's frame count before the lone frame and the
                                        fillers it took
"""
import json, os, struct, sys, time, zlib, fcntl, errno

MAGIC = 0x5442434B  # "KCBT"
PATTERN = bytes((i * 7 + 3) & 0xFF for i in range(4096 - 12))
FILLER = 0xFFFFFFFF


def frame(seq, size=4096):
    body = struct.pack('<II', MAGIC, seq) + PATTERN[:size - 12]
    return body + struct.pack('<I', zlib.crc32(body))


def pingpong(fd, client, n, size, follow=0, run=0):
    """Single frames there and back, each followed at once by `follow` filler frames; returns per-round (rtt, fillers
    sent by this side after a 20 ms wait), the frames of this run seen and the stale ones skipped. Sequence numbers
    carry the run (bits 16-31), so frames left from an earlier run are told apart; fillers are FILLER."""
    m = struct.pack('<I', MAGIC)
    buf = bytearray()
    rounds, fillers, got, stale, last_write = [], 0, 0, 0, time.monotonic()

    def write(seq):
        nonlocal last_write
        while True:
            try:
                os.write(fd, frame(seq, size))
                last_write = time.monotonic()
                return
            except BlockingIOError:
                pass

    def next_seq(deadline):
        # The next data frame of this run (its index), or None at the deadline. A frame is taken only with a good CRC
        # at this run's size, so frames of another size are skipped byte by byte.
        nonlocal buf, fillers, got, stale
        while time.monotonic() < deadline:
            j = buf.find(m)
            if j >= 0 and len(buf) >= j + size:
                f = bytes(buf[j:j + size])
                if zlib.crc32(f[:-4]) != struct.unpack('<I', f[-4:])[0]:
                    del buf[:j + 1]
                    continue
                seq = struct.unpack('<I', f[4:8])[0]
                del buf[:j + size]
                if seq == FILLER:
                    continue
                if seq >> 16 != run:
                    stale += 1
                    continue
                got += 1
                return seq & 0xFFFF
            if j < 0 and len(buf) > 4:
                del buf[:len(buf) - 3]
            try:
                b = os.read(fd, 1 << 20)
                if b:
                    buf += b
                    continue
            except BlockingIOError:
                pass
            if time.monotonic() - last_write > 0.02:
                write(FILLER)
                fillers += 1
        return None

    for i in range(n):
        t0 = time.monotonic()
        f0 = fillers
        if client:
            write(run << 16 | i)
            for _ in range(follow):
                write(FILLER)
            seq = next_seq(t0 + 2.0)
            while seq is not None and seq < i:
                seq = next_seq(t0 + 2.0)
        else:
            seq = next_seq(t0 + 10.0)
            if seq is not None:
                write(run << 16 | seq)
                for _ in range(follow):
                    write(FILLER)
        if seq is None:
            break
        rounds.append(((time.monotonic() - t0) * 1e6, fillers - f0))
    return rounds, got, stale


def ringpos(fd, client, rounds, run, size=112):
    """See `pos` in the usage. Returns the client's rounds as (frames written before the lone one, fillers before it,
    fillers after it, microseconds), or the number of lone frames the server answered."""
    m = struct.pack('<I', MAGIC)
    buf = bytearray()
    ready = run << 16 | 0xFFFE

    def write(seq):
        while True:
            try:
                os.write(fd, frame(seq, size))
                return
            except BlockingIOError:
                pass

    def frames(deadline):
        # Sequence numbers of this run's frames as they arrive, until the deadline (fillers skipped).
        nonlocal buf
        while time.monotonic() < deadline:
            j = buf.find(m)
            if j >= 0 and len(buf) >= j + size:
                f = bytes(buf[j:j + size])
                if zlib.crc32(f[:-4]) != struct.unpack('<I', f[-4:])[0]:
                    del buf[:j + 1]
                    continue
                del buf[:j + size]
                seq = struct.unpack('<I', f[4:8])[0]
                if seq != FILLER and seq >> 16 == run:
                    yield seq & 0xFFFF
                continue
            if j < 0 and len(buf) > 4:
                del buf[:len(buf) - 3]
            try:
                b = os.read(fd, 1 << 20)
                if b:
                    buf += b
            except BlockingIOError:
                pass

    # The client starts: a READY written before the other end opened would wait in its ring (or be lost), so the
    # server answers the client's HELLO (each followed by three fillers, so neither waits).
    hello = run << 16 | 0xFFFD
    if not client:
        answered = 0
        for s in frames(time.monotonic() + 30):
            if s == 0xFFFD:
                write(ready)
            elif s < 0xFFFD:
                write(run << 16 | s)
                answered += 1
            else:
                continue
            for _ in range(3):
                write(FILLER)
            if answered == rounds:
                break
        return answered
    n, deadline = 0, time.monotonic() + 10
    while True:
        write(hello)
        for _ in range(3):
            write(FILLER)
        n += 4
        if any(s == 0xFFFE for s in frames(time.monotonic() + 0.5)):
            break
        if time.monotonic() > deadline:
            return []
    out = []
    for r in range(rounds):
        k = r % 4
        for _ in range(k):
            write(FILLER)
        n += k
        before, t0 = n, time.monotonic()
        write(run << 16 | r)
        n += 1
        extra = 0
        while not any(s == r for s in frames(time.monotonic() + 0.02)):
            if extra == 8:
                break
            write(FILLER)
            n += 1
            extra += 1
        out.append((before, k, extra, round((time.monotonic() - t0) * 1e6)))
    return out


def stats(v):
    v = sorted(v)
    return {'n': len(v), 'p50': v[len(v) // 2], 'p90': v[len(v) * 9 // 10], 'max': v[-1]} if v else {}


def open_dev(dev):
    t = time.monotonic()
    fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd, time.monotonic() - t


def close_dev(fd):
    t = time.monotonic()
    os.close(fd)
    return time.monotonic() - t


def out(**kw):
    kw['host'] = os.uname().nodename
    print(json.dumps(kw), flush=True)


def write_frames(fd, n, spin_limit):
    sent = eagain = 0
    t = time.monotonic()
    while sent < n and time.monotonic() - t < spin_limit:
        try:
            os.write(fd, frame(sent))
            sent += 1
        except BlockingIOError:
            eagain += 1
            time.sleep(0.0005)
    return sent, eagain, time.monotonic() - t


def main():
    cmd, dev = sys.argv[1], sys.argv[2]
    if cmd == 'opentime':
        res = []
        for _ in range(5):
            fd, o = open_dev(dev)
            res.append({'open_ms': o * 1e3, 'close_ms': close_dev(fd) * 1e3})
        out(step='opentime', dev=dev, runs=res)
    elif cmd == 'writer':
        n, hold = int(sys.argv[3]), float(sys.argv[4])
        fd, o = open_dev(dev)
        sent, eagain, w = write_frames(fd, n, 2.0)
        time.sleep(hold)
        out(step='writer', dev=dev, open_ms=o * 1e3, sent=sent, eagain=eagain, write_s=w, close_ms=close_dev(fd) * 1e3)
    elif cmd == 'blast':
        n = int(sys.argv[3])
        fd, o = open_dev(dev)
        sent, eagain, w = write_frames(fd, n, 10.0)
        time.sleep(0.5)
        out(step='blast', dev=dev, open_ms=o * 1e3, sent=sent, eagain=eagain, write_s=w, close_ms=close_dev(fd) * 1e3)
    elif cmd == 'pp':
        client, n, size = sys.argv[3] == 'client', int(sys.argv[4]), int(sys.argv[5])
        follow = int(sys.argv[6]) if len(sys.argv) > 6 else 0
        run = int(sys.argv[7]) if len(sys.argv) > 7 else 0
        fd, o = open_dev(dev)
        if client:
            time.sleep(1.0)
        rounds, got, stale = pingpong(fd, client, n, size, follow, run)
        time.sleep(0.2)
        out(step='pp', side=sys.argv[3], dev=dev, size=size, follow=follow, run=run, rounds=len(rounds), frames_seen=got,
            stale=stale,
            us=stats([r[0] for r in rounds]), fillers=stats([r[1] for r in rounds]),
            fillers_total=sum(r[1] for r in rounds), first_rounds=[(round(r[0]), r[1]) for r in rounds[:12]],
            close_ms=close_dev(fd) * 1e3)
    elif cmd == 'drain':
        # Read whatever the device holds (frames and ends of file of earlier sessions) for SECONDS, then close.
        secs = float(sys.argv[3])
        fd, o = open_dev(dev)
        n = eofs = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < secs:
            try:
                b = os.read(fd, 1 << 20)
                if b:
                    n += len(b)
                else:
                    eofs += 1
            except BlockingIOError:
                time.sleep(0.001)
        out(step='drain', dev=dev, bytes=n, eofs=eofs, close_ms=close_dev(fd) * 1e3)
    elif cmd == 'pos':
        client, rounds, run = sys.argv[3] == 'client', int(sys.argv[4]), int(sys.argv[5])
        fd, o = open_dev(dev)
        if client:
            time.sleep(1.0)
        res = ringpos(fd, client, rounds, run)
        time.sleep(0.2)
        if client:
            by = {}
            for before, k, extra, us in res:
                by.setdefault(before % 4, []).append(extra)
            out(step='pos', side='client', dev=dev, run=run, rounds=len(res),
                fillers_by_position_mod_4={str(p): sorted(set(v)) for p, v in sorted(by.items())}, raw=res,
                close_ms=close_dev(fd) * 1e3)
        else:
            out(step='pos', side='server', dev=dev, run=run, answered=res, close_ms=close_dev(fd) * 1e3)
    elif cmd in ('reader', 'lazy'):
        # reader: sleep DELAY, then open and read; lazy: open, sleep DELAY without reading, then read.
        delay, secs = float(sys.argv[3]), float(sys.argv[4])
        if cmd == 'reader':
            time.sleep(delay)
        fd, o = open_dev(dev)
        if cmd == 'lazy':
            time.sleep(delay)
        buf = bytearray()
        eofs = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < secs:
            try:
                b = os.read(fd, 1 << 20)
            except BlockingIOError:
                time.sleep(0.0005)
                continue
            if not b:
                eofs.append(round((time.monotonic() - t0) * 1e3, 1))
                continue
            buf += b
        intact = damaged = 0
        seqs = []
        i = 0
        m = struct.pack('<I', MAGIC)
        while True:
            j = buf.find(m, i)
            if j < 0:
                break
            f = bytes(buf[j:j + 4096])
            if len(f) == 4096 and zlib.crc32(f[:-4]) == struct.unpack('<I', f[-4:])[0]:
                intact += 1
                seqs.append(struct.unpack('<I', f[4:8])[0])
                i = j + 4096
            else:
                damaged += 1
                i = j + 1
        gaps = sum(1 for a, b in zip(seqs, seqs[1:]) if b != a + 1)
        out(step='reader', dev=dev, open_ms=o * 1e3, bytes=len(buf), frames_intact=intact, frames_damaged=damaged,
            eofs_ms=eofs, first=seqs[0] if seqs else None, last=seqs[-1] if seqs else None, gaps=gaps,
            close_ms=close_dev(fd) * 1e3)


if __name__ == '__main__':
    main()
