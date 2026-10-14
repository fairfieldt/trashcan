#!/usr/bin/env python3
"""Print the six-Mac chain order and node addresses from the live Thunderbolt links, starting at the driver.

usage: ring.py [--driver incite]    ->  NODES=a,b,c,d,e   ADDRS=a.tbN:7071,...   (shell-sourceable)

Each Mac's `tblink links` lists its peers with its own bus; a node's address is resolved on the machine before it in the
chain as <peer>.tb<that machine's bus> (tblink's names). The ring is followed from the driver in one direction: through
its bus-0 link, the same direction as the usual order when the cabling had incite bus 0 -> next. Recabling then needs
no script changes.
"""
import argparse, subprocess, sys

MACS = ["incite", "indite", "inflight", "alight", "invite", "enlight"]


def links(host):
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=8", host, "sudo -n /usr/local/sbin/tblink links"],
                         capture_output=True, text=True).stdout
    peers = {}
    for line in out.splitlines()[1:]:
        f = line.split()
        if len(f) >= 6 and f[5] == "up" and f[1] in MACS:
            peers[int(f[2])] = f[1]
    return peers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--driver", default="incite")
    a = p.parse_args()
    table = {h: links(h) for h in MACS}
    order, addrs, prev, here = [a.driver], [], None, a.driver
    for _ in range(len(MACS) - 1):
        peers = table[here]
        # the first hop leaves the driver through bus 0; later hops take the link that does not go back
        choices = sorted(peers.items()) if prev is None else [(b, h) for b, h in sorted(peers.items()) if h != prev]
        if not choices:
            sys.exit(f"ring broken after {here}: links {peers}")
        bus, nxt = choices[0]
        if nxt in order:
            sys.exit(f"ring closes early at {nxt}: {order}")
        addrs.append(f"{nxt}.tb{bus}:7071")
        order.append(nxt)
        prev, here = here, nxt
    if a.driver not in table[here].values():
        print(f"# warning: {here} has no link back to {a.driver}", file=sys.stderr)
    print(f"NODES={','.join(order[1:])}")
    print(f"ADDRS={','.join(addrs)}")


if __name__ == "__main__":
    main()
