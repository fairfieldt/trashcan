#!/usr/bin/env python3
"""Append incite's netconsole (UDP 6666 from its LAN address) to a log, one timestamped line per datagram."""
import datetime, socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", 6666))
with open(sys.argv[1], "a", buffering=1) as f:
    while True:
        data, addr = s.recvfrom(65536)
        t = datetime.datetime.now().isoformat(timespec="milliseconds")
        for line in data.decode(errors="replace").splitlines():
            f.write(f"{t} {addr[0]} {line}\n")
