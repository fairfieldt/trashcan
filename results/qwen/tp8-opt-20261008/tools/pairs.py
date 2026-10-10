#!/usr/bin/env python3
"""Markdown rows for alternated A/B groups: mean tok/s per side with ranges, the change, GPU wait and exchange per
token, prefill. usage: pairs.py RESULTS_DIR 'LABEL:DEFAULT_PREFIX:CANDIDATE_PREFIX' ..."""
import json, os, re, statistics, sys

def runs(root, prefix):
    out = []
    for name in sorted(os.listdir(root)):
        if not name.endswith('.json') or not re.fullmatch(re.escape(prefix) + r'-\d+\.json', name):
            continue
        for line in open(os.path.join(root, name)):
            line = line.strip()
            if line.startswith('{'):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'tg_tok_s' in r:
                    out.append(r)
    return out

def stat(v, d=2):
    return f"{statistics.fmean(v):.{d}f} ({min(v):.{d}f}-{max(v):.{d}f})" if v else '-'

root = sys.argv[1]
print('| candidate | n | default tok/s | candidate tok/s | change | GPU wait ms/token | exchange ms/token | pp512 tok/s |')
print('|---|---|---|---|---|---|---|---|')
for spec in sys.argv[2:]:
    label, a, b = spec.split(':')
    ra, rb = runs(root, a), runs(root, b)
    ta, tb = [r['tg_tok_s'] for r in ra], [r['tg_tok_s'] for r in rb]
    wa, wb = [r['tg_tp_ms']['wait'] for r in ra], [r['tg_tp_ms']['wait'] for r in rb]
    ea, eb = [r['tg_tp_ms']['exchange'] for r in ra], [r['tg_tp_ms']['exchange'] for r in rb]
    pa, pb = [r['pp_tok_s'] for r in ra], [r['pp_tok_s'] for r in rb]
    ch = (statistics.fmean(tb) / statistics.fmean(ta) - 1) * 100 if ta and tb else float('nan')
    print(f"| {label} | {len(ta)}+{len(tb)} | {stat(ta)} | {stat(tb)} | {ch:+.1f}% | {statistics.fmean(wa):.2f} → {statistics.fmean(wb):.2f} | "
          f"{statistics.fmean(ea):.2f} → {statistics.fmean(eb):.2f} | {statistics.fmean(pa):.0f} → {statistics.fmean(pb):.0f} |")
