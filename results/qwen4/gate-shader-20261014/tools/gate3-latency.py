#!/usr/bin/env python3
"""Pooled release latencies of release-always steps (their raw per-run values), by group of step-name prefixes:
host write to the GPU timestamp after the gate, and to the gate's own clock at the poll that saw the write.

usage: gate3-latency.py DIR NAME=PREFIX[,PREFIX...] [NAME=...]
"""
import json
import pathlib
import sys

d = pathlib.Path(sys.argv[1])


def pct(v, q):
    return v[min(int(len(v) * q), len(v) - 1)]


for arg in sys.argv[2:]:
    name, prefixes = arg.split("=")
    gpu, shader, runs, seen = [], [], 0, 0
    for p in sorted(d.glob("*.json")):
        if not any(p.stem.startswith(x) for x in prefixes.split(",")):
            continue
        text = p.read_text().strip()
        if not text:
            continue
        j = json.loads(text.splitlines()[-1])
        if j.get("step") != "release":
            continue
        runs += j["runs"]
        seen += j["seen"]
        gpu += j["raw_release_to_gpu_past_gate_us"]
        shader += j["raw_release_to_shader_open_us"]
    gpu.sort()
    shader.sort()
    if not gpu:
        print(f"{name}: no runs")
        continue
    line = lambda v: " ".join(f"{k} {pct(v, q):.1f}" for k, q in
                              (("p10", .1), ("p50", .5), ("p90", .9), ("p99", .99))) + f" max {v[-1]:.1f}"
    print(f"{name}: seen {seen}/{runs}; to GPU past gate (us): {line(gpu)}; mean {sum(gpu) / len(gpu):.1f}")
    print(f"{' ' * len(name)}  to the gate's open poll (us): {line(shader)}; mean {sum(shader) / len(shader):.1f}")
