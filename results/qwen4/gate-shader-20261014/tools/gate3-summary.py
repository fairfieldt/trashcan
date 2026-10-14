#!/usr/bin/env python3
"""Summarize gate-check JSON lines (results/gate3/*.json from the round-3 lease scripts) as one table row per step.

usage: gate3-summary.py DIR [DIR...]
"""
import json
import pathlib
import sys


def fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def pct(d, k, nd=1):
    return fmt(d.get(k), nd) if isinstance(d, dict) else "-"


rows = []
for arg in sys.argv[1:]:
    for p in sorted(pathlib.Path(arg).glob("*.json")):
        text = p.read_text().strip()
        if not text:
            rows.append((p.stem, "(no output)"))
            continue
        j = json.loads(text.splitlines()[-1])
        sw = j.get("sweep") or {}
        sweep = f"{sw['kib']}K/{sw['stride']}B/{sw['lines']}" if sw else "-"
        mem = "cached" if j["memory"].startswith("host-cached") else "wc"
        head = f"{p.stem:28} {j['shader']:20} {mem:6} {j['fence']:7} {sweep:16}"
        if j["step"] == "release":
            g, s, f = j["release_to_gpu_past_gate_us"], j["release_to_shader_open_us"], j.get("first_failed_run") or {}
            line = (f"{head} seen {j['seen']}/{j['runs']}"
                    f" | to GPU past gate us p50 {pct(g, 'p50')} p90 {pct(g, 'p90')} p99 {pct(g, 'p99')} max {pct(g, 'max')}"
                    f" | to shader open p50 {pct(s, 'p50')} | polls p50 {pct(j['polls'], 'p50', 0)}"
                    f" | poll load us p50 {pct(j['load_us_mean'], 'p50', 2)}")
            if f:
                line += (f" | failed run: how {f.get('how')} polls {f.get('polls')} {fmt(f.get('gate_ms'), 2)} ms"
                         f" {fmt(f.get('us_per_poll'), 3)} us/poll load {fmt(f.get('load_us_mean'), 2)}"
                         f" [{fmt(f.get('load_us_fewest'), 2)}, {fmt(f.get('load_us_most'), 2)}]"
                         f" sweep {fmt(f.get('sweep_us_mean'), 2)}")
        elif j["step"] == "never":
            line = (f"{head} how {j['how']} ended_by_bound {j['ended_by_bound']} {fmt(j['gate_ms'], 2)} ms"
                    f" polls {j['polls']} {fmt(j['us_per_poll'], 3)} us/poll load {fmt(j.get('load_us_mean'), 2)}"
                    f" [{fmt(j.get('load_us_fewest'), 2)}, {fmt(j.get('load_us_most'), 2)}] sweep {fmt(j.get('sweep_us_mean'), 2)}")
        else:
            line = (f"{head} not_open {j['not_open']}/{j['runs']} | gate GPU us p50 {pct(j['gate_gpu_us'], 'p50', 2)}"
                    f" p99 {pct(j['gate_gpu_us'], 'p99', 2)} | graph GPU us p50 {pct(j['graph_gpu_us'], 'p50', 2)}"
                    f" | started->done host us p50 {pct(j['started_to_done_host_us'], 'p50', 2)}")
        rows.append((p.stem, line))
for _, line in rows:
    print(line)
