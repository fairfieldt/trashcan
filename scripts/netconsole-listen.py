#!/usr/bin/env python3
"""Receive incite's netconsole (UDP 6666) and append each line, timestamped, to a log.

usage: scripts/netconsole-listen.py [LOG]   (default ~/dev/claude-work/trashcan-kbuild/netconsole-incite.log)
"""
import datetime, os, socket, sys

log = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/dev/claude-work/trashcan-kbuild/netconsole-incite.log")
os.makedirs(os.path.dirname(log), exist_ok=True)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", 6666))
with open(log, "a", buffering=1) as f:
    while True:
        data, addr = s.recvfrom(65535)
        now = datetime.datetime.now().isoformat(timespec="milliseconds")
        for line in data.decode(errors="replace").splitlines():
            f.write(f"{now} {addr[0]} {line}\n")
