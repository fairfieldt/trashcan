#!/usr/bin/env python3
"""Flash-Next end to end, main (TCP) against this branch on the streams and over TCP, from window 2's driver JSON
(collected into DIR/incite): decode and prefill tok/s per run and per side, the MTP rates per prompt, and whether the
sides generated the same tokens.

usage: w2_summary.py DIR
"""
import glob
import json
import os
import re
import statistics
import sys

d = os.path.join(sys.argv[1], 'incite')


def load(name):
    p = os.path.join(d, name + '.json')
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p) if l.strip().startswith('{')]
    return rows


def fmt(v):
    return f"{statistics.mean(v):.2f} ({min(v):.2f}-{max(v):.2f})" if len(v) > 1 else (f"{v[0]:.2f}" if v else '-')


bench = {}
for f in sorted(glob.glob(os.path.join(d, 'stream-w2-*.json'))):
    name = os.path.basename(f)[:-5]
    m = re.match(r'stream-w2-(tg|pp2048|pp8192)-(main|stream|tcp)(?:-(\d+))?$', name)
    if not m:
        continue
    r = load(name)
    if not r:
        continue
    r = r[-1]
    bench.setdefault((m.group(1), m.group(2)), []).append((r['pp_tok_s'], r['tg_tok_s']))
    print(f"{name:28s} pp{r['pp']:<5d} {r['pp_tok_s']:7.1f} tok/s   tg{r['tg']:<4d} {r['tg_tok_s']:6.2f} tok/s")
print()
print('| metric | main (TCP) | this branch, streams | this branch, TCP | streams against main |')
print('|---|---|---|---|---|')
for kind, label in (('tg', 'decode, 128 tokens after 512'), ('tg', 'prefill 512'), ('pp2048', 'prefill 2048'),
                    ('pp8192', 'prefill 8192')):
    col = 1 if label.startswith('decode') else 0
    v = {s: [x[col] for x in bench.get((kind, s), [])] for s in ('main', 'stream', 'tcp')}
    gain = f"{100 * (statistics.mean(v['stream']) / statistics.mean(v['main']) - 1):+.2f}%" if v['main'] and v['stream'] else '-'
    print(f"| {label} | {fmt(v['main'])} | {fmt(v['stream'])} | {fmt(v['tcp'])} | {gain} |")

mtp = {}
ids = {}
for f in sorted(glob.glob(os.path.join(d, 'stream-w2-mtp-*.json'))):
    name = os.path.basename(f)[:-5]
    m = re.match(r'stream-w2-mtp-(main|stream|tcp)-(\d+)$', name)
    if not m:
        continue
    for r in load(name) or []:
        mtp.setdefault(m.group(1), []).append(r)
        ids.setdefault(r['prompt'], {}).setdefault(m.group(1), []).append(r['mtp']['ids'])
print()
print('| MTP (3 chat prompts) | main (TCP) | this branch, streams | this branch, TCP | streams against main |')
print('|---|---|---|---|---|')
for key, label in (('mtp', 'speculative tok/s (mean of prompts)'), ('plain', 'plain decode tok/s')):
    v = {}
    for s, rows in mtp.items():
        v[s] = [r['mtp']['tok_s'] if key == 'mtp' else r['plain_tok_s'] for r in rows]
    gain = f"{100 * (statistics.mean(v['stream']) / statistics.mean(v['main']) - 1):+.2f}%" if v.get('main') and v.get('stream') else '-'
    print(f"| {label} | {fmt(v.get('main', []))} | {fmt(v.get('stream', []))} | {fmt(v.get('tcp', []))} | {gain} |")
same = all(len({json.dumps(x) for side in sides.values() for x in side}) == 1 for sides in ids.values())
print(f"\nevery MTP run of a prompt generated the same tokens on every side: {same}")
print(f"every run identical to its plain decode: {all(r['mtp']['identical'] for rows in mtp.values() for r in rows)}")
