#!/usr/bin/env python3
"""Summarize qwen4 stage-bench JSON: per-layer decode ms, phases, and the dispatch profile."""
import json, sys, collections

for path in sys.argv[1:]:
    j = json.load(open(path))
    print(f"== {path}: decode wall {j['decode_wall_ms']:.3f} ms, prefill wall {j['prefill_wall_ms']:.1f} ms")
    for si, st in enumerate(j['decode']):
        lm = st['layer_ms']
        print(f"  stage {si} layers ms: " + " ".join(f"{x:.3f}" for x in lm) + f"  sum {sum(lm):.3f}")
        ph = collections.OrderedDict()
        for li, name, ms in st['phases']:
            ph.setdefault(li, []).append(f"{name} {ms:.3f}")
        for li, v in ph.items():
            print(f"    L{li}: " + ", ".join(v))
    for si, st in enumerate(j['prefill']):
        print(f"  prefill stage {si} layers ms: " + " ".join(f"{x:.1f}" for x in st['layer_ms']))
    for si, disp in enumerate(j.get('dispatches', [])):
        if not disp:
            continue
        print(f"  stage {si} dispatches (label, us, gap before):")
        for lab, ms, gap in disp:
            print(f"    {lab:44s} {ms*1000:8.1f}  gap {gap*1000:6.1f}")
