# Debug the stream lane handshake: python3 greet.py DEV NAME [wait]
import os, struct, sys, threading, time
dev, name = sys.argv[1], sys.argv[2]
t0 = time.time()
def log(*a): print(f"{time.time()-t0:8.3f} {name}", *a, flush=True)
fd = os.open(dev, os.O_RDWR); log("opened")
rfd = os.dup(fd)
heard = threading.Event(); done = threading.Event()
def reader():
    while True:
        b = b""
        while len(b) < 16:
            c = os.read(rfd, 16 - len(b))
            if not c: log("EOF"); done.set(); return
            b += c
        magic, kind, i, n = struct.unpack("<4I", b)
        log("recv", hex(magic), kind, i, n)
        heard.set()
        if kind == 2: done.set(); return
threading.Thread(target=reader, daemon=True).start()
sent = 0
while not heard.is_set():
    os.write(fd, struct.pack("<4I", 0x454e414c, 1, 0, 1)); sent += 1
    time.sleep(0.01)
log("heard after", sent, "hellos; sending READY")
os.write(fd, struct.pack("<4I", 0x454e414c, 2, 0, 1))
done.wait(10); log("done" if done.is_set() else "TIMEOUT waiting for READY")
