# Single-threaded ping-pong over one stream: python3 pingpong.py echo DEV N | python3 pingpong.py ping DEV N SIZE
import os, sys, time
mode, dev, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
size = int(sys.argv[4]) if len(sys.argv) > 4 else 16
fd = os.open(dev, os.O_RDWR)
def rd(k):
    b = b""
    while len(b) < k:
        c = os.read(fd, k - len(b))
        if not c: raise SystemExit("EOF after %d bytes" % len(b))
        b += c
    return b
if mode == "echo":
    hdr = rd(4); size = int.from_bytes(hdr, "little"); os.write(fd, hdr)
    for i in range(n): os.write(fd, rd(size))
    print("echo done", n, flush=True); time.sleep(0.5)
else:
    time.sleep(0.5)
    os.write(fd, size.to_bytes(4, "little")); rd(4)
    buf = bytes(range(256)) * (size // 256 + 1); buf = buf[:size]
    rtt = []
    for i in range(n):
        t = time.perf_counter(); os.write(fd, buf); r = rd(size); rtt.append((time.perf_counter() - t) * 1e6)
        assert r == buf, "corrupt"
    rtt.sort(); print(f"size {size} n {n} rtt_us p50 {rtt[n//2]:.1f} p90 {rtt[n*9//10]:.1f} p99 {rtt[n*99//100]:.1f} max {rtt[-1]:.1f}", flush=True)
