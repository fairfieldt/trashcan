#!/usr/bin/env python3
"""Summarize the six-Mac decode A/B of this directory: tok/s per run and per binary (mean, range), prefill tok/s, and
incite's per-layer GPU milliseconds per decode token.
usage: summary.py [incite/]"""
import glob, json, os, re, statistics, sys

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'incite')
rows = {}
for f in sorted(glob.glob(os.path.join(d, 'dec-*-tg-*.json')) + glob.glob(os.path.join(d, 'dec-*-pp2048-*.json'))):
    try:
        j = json.load(open(f))
    except ValueError:
        continue
    name = os.path.basename(f)[:-5]
    m = re.match(r'dec-(\w+)-(tg|pp2048)-(trashcan-[\w-]+?)(?:-(\d+))?$', name)
    tag, kind, binary = m.group(1), m.group(2), m.group(3)
    gpu = ''
    prof = j.get('decode_profile') or j.get('tg_profile')
    if isinstance(prof, list) and prof:
        gpu = ' incite GPU ms/token ' + ' + '.join(f'{sum(s):.2f}' for s in prof)
    print(f"{name:34s} pp{j['pp']} {j['pp_tok_s']:7.1f} tok/s  tg{j['tg']} {j['tg_tok_s']:6.2f} tok/s{gpu}")
    rows.setdefault((tag, kind, binary), []).append((j['pp_tok_s'], j['tg_tok_s']))
print()
for (tag, kind, binary), v in sorted(rows.items()):
    tg = [x[1] for x in v]
    pp = [x[0] for x in v]
    print(f"{tag} {kind:6s} {binary:18s} n={len(v)}  decode {statistics.mean(tg):6.2f} ({min(tg):.2f}-{max(tg):.2f}) tok/s"
          f"  prefill {statistics.mean(pp):6.1f} tok/s")
