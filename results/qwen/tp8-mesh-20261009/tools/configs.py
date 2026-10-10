#!/usr/bin/env python3
"""Markdown rows per configuration (run name without its repetition number): tok/s of every repetition with the mean,
exchange and GPU wait per generated token, MTP verification per round and tokens per round, prefill.
usage: configs.py RESULTS_DIR 'LABEL:PREFIX' ..."""
import json, os, re, statistics, sys

def runs(root, prefix):
    out = []
    for name in sorted(os.listdir(root)):
        if name.endswith('.json') and re.fullmatch(re.escape(prefix) + r'-\d+\.json', name):
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

root, specs = sys.argv[1], sys.argv[2:]
mtp = any(r.get('mtp') for s in specs for r in runs(root, s.split(':')[1]))
head = '| configuration | n | tok/s (mean, range) | exchange ms/token | GPU wait ms/token |' + (' verification ms/round | tokens/round |' if mtp else '') + ' pp512 tok/s |'
print(head)
print('|' + '---|' * (head.count('|') - 1))
for spec in specs:
    label, prefix = spec.rsplit(':', 1)
    rs = runs(root, prefix)
    tg = [r['tg_tok_s'] for r in rs]
    f = lambda k: statistics.fmean(r['tg_tp_ms'][k] for r in rs)
    row = f"| {label} | {len(rs)} | {statistics.fmean(tg):.2f} ({min(tg):.2f}-{max(tg):.2f}) | {f('exchange'):.2f} | {f('wait'):.2f} |"
    if mtp:
        row += f" {statistics.fmean(r['mtp']['verify_ms_per_round'] for r in rs):.1f} | {statistics.fmean(r['mtp']['tokens_per_round'] for r in rs):.2f} |"
    row += f" {statistics.fmean(r['pp_tok_s'] for r in rs):.0f} |"
    print(row)
