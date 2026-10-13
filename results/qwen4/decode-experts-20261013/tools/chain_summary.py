#!/usr/bin/env python3
"""Summarize a six-Mac decode A/B (gather.sh first): tok/s per run and per binary (mean, range) and prefill tok/s.
usage: chain_summary.py [incite/]"""
import glob, json, os, re, statistics, sys

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'incite')
rows = {}
for f in sorted(glob.glob(os.path.join(d, 'moe-*-tg-*.json')) + glob.glob(os.path.join(d, 'moe-*-pp2048-*.json'))):
    try:
        j = json.load(open(f))
    except ValueError:
        continue
    name = os.path.basename(f)[:-5]
    m = re.match(r'moe-(\w+)-(tg|pp2048)-(base|new)(?:-(\d+))?$', name)
    if not m:
        continue
    tag, kind, binary = m.group(1), m.group(2), m.group(3)
    lay = j.get('decode_layer_ms')
    extra = f"  decode layer ms {lay:.3f}" if isinstance(lay, (int, float)) else ''
    print(f"{name:26s} pp{j['pp']} {j['pp_tok_s']:7.1f} tok/s  tg{j['tg']} {j['tg_tok_s']:6.2f} tok/s{extra}")
    rows.setdefault((tag, kind, binary), []).append((j['pp_tok_s'], j['tg_tok_s']))
print()
for (tag, kind, binary), v in sorted(rows.items()):
    tg = [x[1] for x in v]
    pp = [x[0] for x in v]
    print(f"{tag} {kind:6s} {binary:5s} n={len(v)}  decode {statistics.mean(tg):6.2f} ({min(tg):.2f}-{max(tg):.2f}) tok/s"
          f"  prefill {statistics.mean(pp):6.1f} ({min(pp):.1f}-{max(pp):.1f}) tok/s")
