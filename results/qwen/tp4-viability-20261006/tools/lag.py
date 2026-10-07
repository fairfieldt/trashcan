# Does a stream frame wait for the next one? python3 lag.py send DEV N GAP | python3 lag.py recv DEV SECONDS
import os, struct, sys, threading, time
mode, dev = sys.argv[1], sys.argv[2]
fd = os.open(dev, os.O_RDWR)
t0 = time.time()
if mode == "send":
    n, gap = int(sys.argv[3]), float(sys.argv[4])
    time.sleep(1.0)
    for k in range(n):
        os.write(fd, struct.pack("<Id", k, time.time()))
        print(f"{time.time()-t0:7.3f} sent {k}", flush=True)
        time.sleep(gap)
    time.sleep(2.0)
    print(f"{time.time()-t0:7.3f} closing", flush=True)
else:
    end = time.time() + float(sys.argv[3])
    def rd():
        while True:
            b = os.read(fd, 12)
            if not b:
                print(f"{time.time()-t0:7.3f} EOF", flush=True); return
            k, ts = struct.unpack("<Id", b)
            print(f"{time.time()-t0:7.3f} got {k} sent {ts-t0:7.3f} late {1e3*(time.time()-ts):7.1f} ms", flush=True)
    threading.Thread(target=rd, daemon=True).start()
    while time.time() < end: time.sleep(0.1)
