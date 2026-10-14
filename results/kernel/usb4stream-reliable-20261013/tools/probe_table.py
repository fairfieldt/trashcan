#!/usr/bin/env python3
"""The raw ping-pong probe (tbcheck.py pp) as a table: one line per link (client host) and variant.

usage: probe_table.py PROBE.jsonl...
"""
import json, sys

for path in sys.argv[1:]:
    rows = {}
    for line in open(path):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("side") == "client":
            rows[(d["host"], d["follow"])] = d
    print(f"== {path}: {len(rows)} client runs")
    for (h, f), d in sorted(rows.items()):
        u = d.get("us", {})
        print(f"  {h:9s} follow {f}: {d['rounds']:3d} rounds, round trip p50 {u.get('p50', 0):9.0f} us p90 "
              f"{u.get('p90', 0):9.0f} us, fillers p50 {d['fillers'].get('p50')} total {d['fillers_total']}")
