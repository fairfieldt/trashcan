#!/usr/bin/env python3
"""Bounded test traffic for ethd: unicast EtherType 0x88b5 frames between two given addresses, nothing else.
Needs root (a packet socket).

  peer.py echo IFACE PEER_MAC [--count N] [--idle S]
      Sends back every such frame that comes from PEER to IFACE's address, with the addresses swapped; stops after
      N echoes or S idle seconds. The peer for `ethd selftest`.
  peer.py send IFACE PEER_MAC [--count N] [--size B] [--window W]
      Sends N numbered frames from IFACE to PEER (W in flight at most) and checks that each comes back intact from
      an echoing peer. Used on ethd's TAP interface."""
import argparse, os, socket, sys, time

ETH_P = 0x88B5
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["echo", "send"])
ap.add_argument("iface")
ap.add_argument("peer")
ap.add_argument("--count", type=int, default=1000)
ap.add_argument("--idle", type=float, default=10.0)
ap.add_argument("--size", type=int, default=1514)
ap.add_argument("--window", type=int, default=32)
a = ap.parse_args()
ours = bytes.fromhex(open(f"/sys/class/net/{a.iface}/address").read().strip().replace(":", ""))
peer = bytes.fromhex(a.peer.replace(":", ""))
assert not peer[0] & 1, "the peer must be a unicast address"
s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P))
s.bind((a.iface, ETH_P))
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
s.settimeout(0.2)

if a.mode == "echo":
    n, ignored, last, first = 0, 0, time.time(), None
    while n < a.count and time.time() - last < a.idle:
        try:
            f = s.recv(2048)
        except socket.timeout:
            continue
        if f[:6] != ours or f[6:12] != peer:
            ignored += 1
            continue
        s.send(peer + ours + f[12:])
        n += 1
        last = time.time()
        first = first or last
    print(f"echoed {n} frames from {a.peer} in {(last - first) if first else 0:.3f} s, ignored {ignored}")
    sys.exit(0)

tag = os.urandom(4)
pattern = bytes(range(256)) * 8
def frame(i):
    # The payload is a byte ramp starting at a frame-dependent offset, so corruption and mix-ups show.
    head = peer + ours + ETH_P.to_bytes(2, "big") + b"ethd-peer" + tag + i.to_bytes(4, "big")
    return head + pattern[i % 251 : i % 251 + a.size - len(head)]
sent = got = bad = 0
seen = set()
start = last = time.time()
while got < a.count and time.time() - last < a.idle:
    while sent < a.count and sent - got < a.window:
        s.send(frame(sent))
        sent += 1
    try:
        f = s.recv(2048)
    except socket.timeout:
        continue
    if f[:6] != ours or f[6:12] != peer or f[14:23] != b"ethd-peer" or f[23:27] != tag:
        continue
    i = int.from_bytes(f[27:31], "big")
    want = frame(i)
    if i >= a.count or i in seen or f[12:] != want[12:]:
        bad += 1
        continue
    seen.add(i)
    got += 1
    last = time.time()
secs = last - start
print(f"sent {sent}, echoed {got}, corrupt {bad}, in {secs:.3f} s, {got / secs:.0f} frames/s each way")
sys.exit(0 if got == a.count and bad == 0 else 1)
