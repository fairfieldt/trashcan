#!/usr/bin/env python3
"""Markdown rows for alternated MTP A/B groups. usage: mtppairs.py RESULTS_DIR 'LABEL:DEFAULT_PREFIX:CANDIDATE_PREFIX' ..."""
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
                if 'tg_tok_s' in r and r.get('mtp'):
                    out.append(r)
    return out

def stat(v, d=2):
    return f"{statistics.fmean(v):.{d}f} ({min(v):.{d}f}-{max(v):.{d}f})" if v else '-'

root = sys.argv[1]
print('| candidate | n | default tok/s | candidate tok/s | change | verification ms/round | tokens/round |')
print('|---|---|---|---|---|---|---|')
for spec in sys.argv[2:]:
    label, a, b = spec.split(':')
    ra, rb = runs(root, a), runs(root, b)
    ta, tb = [r['tg_tok_s'] for r in ra], [r['tg_tok_s'] for r in rb]
    va, vb = [r['mtp']['verify_ms_per_round'] for r in ra], [r['mtp']['verify_ms_per_round'] for r in rb]
    ka, kb = [r['mtp']['tokens_per_round'] for r in ra], [r['mtp']['tokens_per_round'] for r in rb]
    ch = (statistics.fmean(tb) / statistics.fmean(ta) - 1) * 100 if ta and tb else float('nan')
    print(f"| {label} | {len(ta)}+{len(tb)} | {stat(ta)} | {stat(tb)} | {ch:+.1f}% | {statistics.fmean(va):.1f} → {statistics.fmean(vb):.1f} | "
          f"{statistics.fmean(ka):.2f} → {statistics.fmean(kb):.2f} |")
