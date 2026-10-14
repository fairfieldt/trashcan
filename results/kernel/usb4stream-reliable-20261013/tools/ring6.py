#!/usr/bin/env python3
"""The six ring links from the live Thunderbolt links, as results/qwen4/chain-20261012/tools/ring.py follows them
(from incite through its bus-0 link), closing link included: one line per link, `MAC NEXT ADDR` with ADDR the next
Mac's tblink name as MAC resolves it (<next>.tb<MAC's bus>). Exits non-zero unless the six links make one ring with
every link up.

usage: ring6.py [--start incite]
"""
import argparse, subprocess, sys

MACS = ["incite", "indite", "inflight", "alight", "invite", "enlight"]


def links(host):
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=8", host, "/usr/local/sbin/tblink links"],
                         capture_output=True, text=True).stdout
    peers = {}
    for line in out.splitlines()[1:]:
        f = line.split("\t")
        if len(f) >= 6 and f[5] == "up" and f[1] in MACS:
            peers[int(f[2])] = f[1]
    return peers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="incite")
    a = p.parse_args()
    table = {h: links(h) for h in MACS}
    order, prev, here = [a.start], None, a.start
    out = []
    while True:
        peers = table[here]
        choices = sorted(peers.items()) if prev is None else [(b, h) for b, h in sorted(peers.items()) if h != prev]
        if not choices:
            sys.exit(f"ring broken after {here}: links {peers}")
        bus, nxt = choices[0]
        out.append(f"{here} {nxt} {nxt}.tb{bus}")
        if nxt == a.start:
            break
        if nxt in order:
            sys.exit(f"ring closes early at {nxt}: {order}")
        order.append(nxt)
        prev, here = here, nxt
    if len(out) != len(MACS):
        sys.exit(f"ring of {len(out)} links, not {len(MACS)}: {out}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
