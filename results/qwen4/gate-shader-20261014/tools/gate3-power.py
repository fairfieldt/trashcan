#!/usr/bin/env python3
"""GPU power during the lease-2 power phases: alight's SMC samples (system-crash-capture telemetry, one per second,
from the netconsole receiver's log) cut into the windows the lease log and the busy steps' JSON give. The first and
last second of each window are dropped.

usage: gate3-power.py NETCONSOLE_LOG LEASE_LOG RESULTS_DIR [HOST]
"""
import datetime
import json
import pathlib
import re
import statistics
import sys

netlog, leaselog, results = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
host = sys.argv[4] if len(sys.argv) > 4 else "alight"

windows = []
idle = {}
for line in open(leaselog):
    m = re.match(r"(\S+): idle (from|to) ([0-9.]+)", line)
    if m:
        idle.setdefault(m.group(1), {})[m.group(2)] = float(m.group(3))
for name, w in idle.items():
    windows.append((name, w["from"], w["to"]))
for p in sorted(results.glob("*-power-*.json")):
    j = json.loads(p.read_text().strip().splitlines()[-1])
    windows.append((p.stem, j["utc_start"], j["utc_end"]))
windows.sort(key=lambda w: w[1])

samples = []
for line in open(netlog, errors="replace"):
    if "CRASHCAPTURE" not in line or f'"host":"{host}"' not in line:
        continue
    try:
        j = json.loads(line[line.index("{"):])
    except ValueError:
        continue
    if j.get("event") != "telemetry":
        continue
    t = datetime.datetime.fromisoformat(j["utc"]).timestamp()
    s = j["smc"]
    try:
        samples.append((t, s["PG0R"]["watts"], s["PG1R"]["watts"], s["PSTR"]["watts"], s["PCPC"]["watts"],
                        s["TG0D"]["celsius"]))
    except KeyError:
        continue

print(f"{'window':24} {'s':>5} {'n':>3} {'PG0R W':>14} {'PG1R W':>7} {'PSTR W':>7} {'PCPC W':>7} {'TG0D C':>7}")
for name, a, b in windows:
    got = [x for x in samples if a + 1 <= x[0] <= b - 1]
    if not got:
        print(f"{name:24} {b - a:5.1f}   0 (no samples)")
        continue
    col = lambda i: [x[i] for x in got]
    pg0 = col(1)
    print(f"{name:24} {b - a:5.1f} {len(got):3} {statistics.mean(pg0):6.1f} [{min(pg0):4.1f}-{max(pg0):4.1f}]"
          f" {statistics.mean(col(2)):7.1f} {statistics.mean(col(3)):7.1f} {statistics.mean(col(4)):7.1f}"
          f" {max(col(5)):7.1f}")
