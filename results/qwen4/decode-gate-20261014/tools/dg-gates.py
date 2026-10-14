#!/usr/bin/env python3
"""The gates' statistics of decode-gate runs: the driver's (`"gate"` in bench JSON, or the `gate: {...}` line validate
prints on stderr) and every node's (`qwen4 node: gate: {...}` at the end of its session). One line per process and
stage: graphs released, open at the first poll, timeouts, release latency (host write to the GPU past the gate, us),
the GPU cost of an open gate (us), waits (ms) and the poll loads' shader-clock ticks (550-800: host memory).

usage: dg-gates.py FILE...   (*.out, *.err, *-node.log)
"""
import json
import sys


def stats_of(path):
    found = []
    for line in open(path, errors="replace"):
        line = line.strip()
        if line.startswith("{"):
            try:
                j = json.loads(line)
            except ValueError:
                continue
            if isinstance(j, dict) and "gate" in j:
                found.append(j["gate"])
        for marker in ("qwen4 node: gate: ", "gate: "):
            if line.startswith(marker):
                try:
                    found.append(json.loads(line[len(marker):]))
                except ValueError:
                    pass
                break
    return found


def p(d, k, nd=1):
    return "-" if not d else f"{d[k]:.{nd}f}"


total = 0
for path in sys.argv[1:]:
    for g in stats_of(path):
        total += g.get("timeouts", 0)
        for s in g.get("stages", []):
            st = s["stats"]
            r, o, w, t = (st.get("release_to_past_gate_us"), st.get("open_gate_gpu_us"), st.get("wait_ms"),
                          st.get("poll_load_ticks"))
            print(f"{path.split('/')[-1]:34} layers {s['layers']} gated {s['gated']}: released {st['released']}, "
                  f"open at start {st['open_at_start']}, timeouts {st['timeouts']}, abandoned {st['abandoned']} | "
                  f"release us p50 {p(r, 'p50')} p90 {p(r, 'p90')} p99 {p(r, 'p99')} max {p(r, 'max')} | "
                  f"open gate us p50 {p(o, 'p50', 2)} | wait ms p50 {p(w, 'p50', 2)} | poll load ticks p50 {p(t, 'p50', 0)}")
print(f"gate timeouts reported by the processes: {total}")
