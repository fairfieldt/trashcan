#!/usr/bin/env python3
"""Decode rates of dg-measure.sh: per layout (one process on both GPUs, or a node and a driver on one GPU each; the
driver's GPU) the tok/s of `bench` (512 decode steps) with gates (on) and without (off) in each alternated round, the
gain of on over off, and the gates' release latency (host write of the ticket to the GPU timestamp past the gate,
calibrated timestamps) and waits, from the driver's bench JSON and the node's log.

usage: dg-bench.py RESULTS_DIR
"""
import json
import pathlib
import re
import statistics
import sys

d = pathlib.Path(sys.argv[1])


def bench(path):
    for line in path.read_text().splitlines():
        if line.startswith("{"):
            try:
                j = json.loads(line)
            except ValueError:
                continue
            if "tg_tok_s" in j:
                return j
    return None


def node_gate(path):
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if "qwen4 node: gate: " in line:
            return json.loads(line[line.index("{"):])
    return None


def stats(gates):
    """Release latency and waits of every stage of every process (gate JSON objects)."""
    out = []
    for g in gates:
        for s in (g or {}).get("stages", []):
            st = s["stats"]
            r, w = st.get("release_to_past_gate_us"), st.get("wait_ms")
            if st["released"]:
                out.append(f"layers {s['layers'][0]}..{s['layers'][1]}: {st['released']} released, {st['timeouts']} "
                           f"timeouts, release us p50 {r['p50']:.1f} p90 {r['p90']:.1f} p99 {r['p99']:.1f}" if r else
                           f"layers {s['layers']}: {st['released']} released, {st['timeouts']} timeouts")
                if w:
                    out[-1] += f", wait ms p50 {w['p50']:.2f}"
    return "; ".join(out)


rows = {}
for p in sorted(d.glob("bench-*.out")):
    m = re.match(r"bench-(one|two)-(\d)-(on|off)-(\d)\.out", p.name)
    if not m:
        continue
    layout, g, arm, r = m.groups()
    j = bench(p)
    if not j:
        print(f"{p.name}: no result")
        continue
    node = node_gate(d / p.name.replace(".out", "-node.log"))
    rows.setdefault((layout, g), {}).setdefault(r, {})[arm] = (j["tg_tok_s"], stats([j.get("gate"), node]))
for (layout, g), rounds in sorted(rows.items()):
    what = "one process, GPUs %s,%s" % (g, 1 - int(g)) if layout == "one" else "node on GPU %d, driver on GPU %s" % (1 - int(g), g)
    print(f"== {what}")
    gains = []
    for r, arms in sorted(rounds.items()):
        on, off = arms.get("on"), arms.get("off")
        if on and off:
            gains.append(on[0] / off[0] - 1)
        print(f"  round {r}: on {on[0] if on else float('nan'):.2f} tok/s, off {off[0] if off else float('nan'):.2f} "
              f"tok/s{f', gain {100 * (on[0] / off[0] - 1):+.2f}%' if on and off else ''}")
        if on:
            print(f"    gates: {on[1]}")
    if gains:
        print(f"  mean gain {100 * statistics.mean(gains):+.2f}% over {len(gains)} pairs")
