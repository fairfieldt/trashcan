#!/usr/bin/env python3
"""Per-token host phases of alternated decode runs: mean tok/s and the milliseconds per generated token the driver
spent waiting for the GPUs and in each part of the all-reduces (the new `encode`, `decode`, `wire`, `sent` and
`first` fields of trashcan-r2 and later). usage: phases.py RESULTS_DIR PREFIX..."""
import json, os, re, statistics, sys

def runs(root, prefix):
    out = []
    for name in sorted(os.listdir(root)):
        if not re.fullmatch(re.escape(prefix) + r'-\d+\.json', name):
            continue
        for line in open(os.path.join(root, name)):
            if line.startswith('{'):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'tg_tok_s' in r:
                    out.append(r)
    return out

root = sys.argv[1]
keys = ['wait', 'exchange', 'sum', 'total', 'encode', 'decode', 'wire', 'sent', 'first']
print('| configuration | n | tok/s | pp512 | ' + ' | '.join(keys) + ' |')
print('|---|---|---|---|' + '---|' * len(keys))
for prefix in sys.argv[2:]:
    rs = runs(root, prefix)
    if not rs:
        continue
    tg = [r['tg_tok_s'] for r in rs]
    pp = [r['pp_tok_s'] for r in rs]
    cells = []
    for k in keys:
        v = [r['tg_tp_ms'].get(k) for r in rs if r['tg_tp_ms'].get(k) is not None]
        cells.append(f"{statistics.fmean(v):.2f}" if v else '-')
    mtp = [r['mtp']['verify_ms_per_round'] for r in rs if r.get('mtp')]
    extra = f" verify {statistics.fmean(mtp):.1f} ms/round" if mtp else ''
    print(f"| {prefix}{extra} | {len(tg)} | {statistics.fmean(tg):.2f} ({min(tg):.2f}-{max(tg):.2f}) | {statistics.fmean(pp):.0f} | " + ' | '.join(cells) + ' |')
