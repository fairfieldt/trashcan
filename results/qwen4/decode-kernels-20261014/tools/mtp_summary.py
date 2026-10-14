#!/usr/bin/env python3
"""Summarize `qwen4 mtp spec` runs (one JSON line per run and prompt): decode rates with and without drafts per
prompt (means over runs), tokens per round and milliseconds per round. Session 3's lines hold the adaptive rounds
under "mtp" and fixed rounds under "fixed"; earlier sessions' lines hold one speculative pass at the top level.

usage: summary.py RUN.json...
"""
import json, statistics, sys


def passes(r):
    if 'mtp' not in r:
        return {'mtp': {'tok_s': r['mtp_tok_s'], 'identical': r['identical'], **r}}
    return {k: r[k] for k in ('mtp', 'pooled', 'fixed') if k in r}


for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if not rows:
        continue
    labels = list(passes(rows[0]))
    same = all(p['identical'] for r in rows for p in passes(r).values())
    print(f"== {f.split('/')[-1]}: k up to {rows[0]['draft_tokens']}, draft vocabulary {rows[0]['draft_vocab']}, "
          f"{len({r['run'] for r in rows})} runs, all identical: {same}")
    print("prompt\tpass\tplain tok/s\tMTP tok/s\tgain\ttokens/round\tdraft ms\tverify ms\trollback ms\trounds by k\tshadow acc\tverify ms by k")
    means = {l: ([], []) for l in labels}
    for p in dict.fromkeys(r['prompt'] for r in rows):
        rs = [r for r in rows if r['prompt'] == p]
        pl = statistics.mean(r['plain_tok_s'] for r in rs)
        for l in labels:
            ps = [passes(r)[l] for r in rs]
            mt = statistics.mean(x['tok_s'] for x in ps)
            means[l][0].append(pl)
            means[l][1].append(mt)
            tpr = statistics.mean(x['tokens_per_round'] for x in ps)
            ms = {k: statistics.mean(x['ms_per_round'][k] for x in ps) for k in ('draft', 'verify', 'rollback')}
            byk = [sum(x['stats'].get('by_k', [0, 0, 0, 0])[i] for x in ps) for i in range(4)]
            sh = sum(x['stats'].get('shadow_hits', 0) for x in ps) / max(byk[0], 1)
            vk = [sum(x['stats'].get('verify_ms_by_k', [0] * 4)[i] for x in ps) / max(byk[i], 1) for i in range(4)]
            print(f"{p}\t{l}\t{pl:.2f}\t{mt:.2f}\t{mt / pl - 1:+.1%}\t{tpr:.2f}\t{ms['draft']:.1f}\t{ms['verify']:.1f}"
                  f"\t{ms['rollback']:.2f}\t{'/'.join(map(str, byk))}\t{sh:.2f}\t{'/'.join(f'{v:.0f}' for v in vk)}")
    for l, (a, b) in means.items():
        pl, mt = statistics.mean(a), statistics.mean(b)
        print(f"mean\t{l}\t{pl:.2f}\t{mt:.2f}\t{mt / pl - 1:+.1%}")
