#!/usr/bin/env python3
"""MTP runs (`qwen4 mtp spec`, one JSON line per prompt and run): plain and speculative tok/s per prompt and variant,
whether the ids were identical, and the chat mean (the mean of the three chat prompts' rates).
usage: mtp_summary.py DIR PATTERN    e.g. mtp_summary.py . 'host-s3-mtp-*'"""
import glob, json, os, re, statistics, sys
from collections import defaultdict

d, pattern = sys.argv[1:3]
rates = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f'{d}/incite/{pattern}.json')):
    variant = re.sub(r'-\d+$', '', os.path.basename(f)[:-5])
    for line in open(f):
        line = line.strip()
        if not line.startswith('{'):
            continue
        j = json.loads(line)
        if 'mtp' not in j:
            continue
        prompt = os.path.basename(str(j.get('prompt', '?'))).removesuffix('.ids')
        m = j['mtp']
        rates[variant][prompt].append((j['plain_tok_s'], m['tok_s'], m.get('identical'), m.get('tokens_per_round')))
for v, by in rates.items():
    plain = [statistics.mean(r[0] for r in rs) for rs in by.values()]
    spec = [statistics.mean(r[1] for r in rs) for rs in by.values()]
    for p, rs in by.items():
        print(f"{v:22s} {p:12s} plain {statistics.mean(r[0] for r in rs):6.2f}  mtp {statistics.mean(r[1] for r in rs):6.2f} tok/s"
              f"  tokens/round {statistics.mean(r[3] for r in rs):.2f}  identical {all(r[2] for r in rs)}  runs {len(rs)}")
    print(f"{v:22s} {'chat mean':12s} plain {statistics.mean(plain):6.2f}  mtp {statistics.mean(spec):6.2f} tok/s")
