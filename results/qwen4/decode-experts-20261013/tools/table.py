#!/usr/bin/env python3
"""Compact per-dispatch table of the MoE phase over several stage-bench runs (mean over layers 5, 6, 7 of stage 1).
usage: table.py [--chunk] RUN.json...   (--chunk: the profiled chunk of --batch tokens instead of decode steps)"""
import json, sys, os, collections
sys.path.insert(0, os.path.dirname(__file__))
from prof import moe_rows  # noqa: E402

chunk = '--chunk' in sys.argv
paths = [a for a in sys.argv[1:] if not a.startswith('--')]
for path in paths:
    j = json.load(open(path))
    key = 'prefill_dispatches' if chunk else 'dispatches'
    d = j.get(key) or []
    if len(d) < 2:
        print(f'{os.path.basename(path)}: no profile'); continue
    rows = moe_rows(d[1])
    names = collections.OrderedDict()
    for lab, us, gap in rows:
        short = lab.split(' ')[0].replace('qwen4_', '').replace('qwen_', '')
        if short in ('combine', 'copy'):
            continue
        names[short] = names.get(short, 0) + us
    tot = sum(names.values())
    ph = j['prefill' if chunk else 'decode'][1]['phases']
    exp = sum(ms for li, n, ms in ph if n == 'experts') / 4
    rt = sum(ms for li, n, ms in ph if n == 'router') / 4
    print(f'{os.path.basename(path)[:28]:28s} MoE {tot:6.1f} us | ' + '  '.join(f'{k} {v:6.1f}' for k, v in names.items())
          + f' | phases router {rt * 1000:5.1f} experts {exp * 1000:6.1f} | wall {j["decode_wall_ms" if not chunk else "prefill_wall_ms"]:.2f}')
