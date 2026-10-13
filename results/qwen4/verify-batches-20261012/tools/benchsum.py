#!/usr/bin/env python3
"""Six-Mac decode A/B summary: tok/s per run and per binary (mean, range), prefill tok/s, and incite's per-layer GPU
milliseconds per decode token, for files NAME-tg-BINARY-RUN.json.
usage: benchsum.py JSON..."""
import json, os, re, statistics, sys
rows = {}
for f in sys.argv[1:]:
    try:
        j = json.load(open(f))
    except ValueError:
        print(f"{f}: unreadable")
        continue
    name = os.path.basename(f)[:-5]
    m = re.match(r'(.*)-tg-(trashcan-[\w]+?)-(\d+)$', name)
    if not m:
        continue
    binary = m.group(2)
    prof = j.get('decode_layer_ms')
    gpu = ''
    if isinstance(prof, list) and prof:
        gpu = ' incite GPU ms/token ' + ' + '.join(f'{sum(s):.2f}' for s in prof)
    print(f"{name:40s} pp{j['pp']} {j['pp_tok_s']:7.1f} tok/s  tg{j['tg']} {j['tg_tok_s']:6.2f} tok/s{gpu}")
    rows.setdefault(binary, []).append((j['pp_tok_s'], j['tg_tok_s']))
print()
for binary, v in sorted(rows.items()):
    tg = [x[1] for x in v]
    pp = [x[0] for x in v]
    print(f"{binary:20s} n={len(v)}  decode {statistics.mean(tg):6.2f} ({min(tg):.2f}-{max(tg):.2f}) tok/s"
          f"  prefill {statistics.mean(pp):6.1f} tok/s")
