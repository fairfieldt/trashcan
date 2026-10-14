#!/usr/bin/env python3
"""Summarize `qwen4 mtp spec` runs (one JSON line per run and prompt): plain and speculative decode rates (means over
runs), identity, acceptance of drafts (rounds with drafts, and shadow drafts during plain steps), rounds by drafts,
milliseconds per round by drafts and per draft step.

usage: summary.py RUN.json...
"""
import json, statistics, sys

for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f) if l.strip().startswith('{')]
    if not rows:
        continue
    same = all(r['mtp']['identical'] for r in rows)
    s = rows[0].get('sampling', {'temperature': 0})
    t = s.get('temperature', 0)
    print(f"== {f.split('/')[-1]}: T {t}" + (f" top-k {s['top_k']} top-p {s['top_p']:.2f}" if t else '') +
          f", {len({r['run'] for r in rows})} runs, all identical: {same}")
    print('prompt\tplain tok/s\tMTP tok/s\tgain\taccepted/draft\tshadow\ttokens/round\trounds by k\tms by k\tdraft ms/step'
          '\tdropped (hits)')
    means = ([], [])
    for p in dict.fromkeys(r['prompt'] for r in rows):
        rs = [r for r in rows if r['prompt'] == p]
        pl = statistics.mean(r['plain_tok_s'] for r in rs)
        mt = statistics.mean(r['mtp']['tok_s'] for r in rs)
        means[0].append(pl)
        means[1].append(mt)
        st = [r['mtp']['stats'] for r in rs]
        drafted, accepted = sum(x['drafted'] for x in st), sum(x['accepted'] for x in st)
        byk = [sum(x['by_k'][i] for x in st) for i in range(4)]
        sh = sum(x['shadow_hits'] for x in st) / max(byk[0], 1)
        vk = [sum(x['verify_ms_by_k'][i] for x in st) / max(byk[i], 1) for i in range(4)]
        dropped = sum(x.get('dropped', 0) for x in st)
        steps = drafted + dropped
        dms = sum(x['draft_ms'] for x in st) / max(steps, 1)
        tpr = statistics.mean(r['mtp']['tokens_per_round'] for r in rs)
        dh = sum(x.get('dropped_hits', 0) for x in st)
        print(f"{p}\t{pl:.2f}\t{mt:.2f}\t{mt / pl - 1:+.1%}\t{accepted / max(drafted, 1):.3f}\t{sh:.3f}\t{tpr:.2f}"
              f"\t{'/'.join(map(str, byk))}\t{'/'.join(f'{v:.1f}' for v in vk)}\t{dms:.2f}\t{dropped} ({dh})")
    pl, mt = statistics.mean(means[0]), statistics.mean(means[1])
    print(f"mean\t{pl:.2f}\t{mt:.2f}\t{mt / pl - 1:+.1%}")
