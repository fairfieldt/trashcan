#!/usr/bin/env python3
"""Summarize `qwen4 mtp spec` runs (one JSON line per run and prompt): decode rates with and without drafts per
prompt (means over runs), tokens per round and milliseconds per round.

usage: summary.py RUN.json...
"""
import json, statistics, sys

for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if not rows:
        continue
    print(f"== {f.split('/')[-1]}: k={rows[0]['draft_tokens']}, draft vocabulary {rows[0]['draft_vocab']}, "
          f"{len({r['run'] for r in rows})} runs, all identical: {all(r['identical'] for r in rows)}")
    print("prompt\tplain tok/s\tMTP tok/s\tgain\ttokens/round\tdraft ms\tverify ms\trollback ms")
    allp, allm = [], []
    for p in dict.fromkeys(r['prompt'] for r in rows):
        rs = [r for r in rows if r['prompt'] == p]
        pl = statistics.mean(r['plain_tok_s'] for r in rs)
        mt = statistics.mean(r['mtp_tok_s'] for r in rs)
        allp.append(pl)
        allm.append(mt)
        tpr = statistics.mean(r['tokens_per_round'] for r in rs)
        ms = {k: statistics.mean(r['ms_per_round'][k] for r in rs) for k in ('draft', 'verify', 'rollback')}
        print(f"{p}\t{pl:.2f}\t{mt:.2f}\t{mt / pl - 1:+.1%}\t{tpr:.2f}\t{ms['draft']:.1f}\t{ms['verify']:.1f}\t{ms['rollback']:.2f}")
    pl, mt = statistics.mean(allp), statistics.mean(allm)
    print(f"mean\t{pl:.2f}\t{mt:.2f}\t{mt / pl - 1:+.1%}")
