#!/usr/bin/env python3
"""Prints (and appends to FILE, if given) the UDP datagrams ethd's netlog mode broadcasts: one log line each.

usage: netlog-listen.py [--port 6666] [FILE]
"""
import argparse
import socket
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=6666)
ap.add_argument("file", nargs="?")
args = ap.parse_args()
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
if hasattr(socket, "SO_REUSEPORT"):
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
s.bind(("", args.port))
out = open(args.file, "a", buffering=1) if args.file else None
while True:
    data, (host, _) = s.recvfrom(65536)
    for line in data.decode(errors="replace").splitlines():
        text = f"{time.strftime('%H:%M:%S')} {host} {line}"
        print(text, flush=True)
        if out:
            out.write(text + "\n")
