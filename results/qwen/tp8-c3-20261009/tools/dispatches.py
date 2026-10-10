#!/usr/bin/env python3
"""Per-dispatch GPU times of a TRASHCAN_TP_PROFILE_DISPATCH=1 trace (profile-*.jsonl[.gz]): for the steps with
N columns, the median microseconds of each labelled dispatch (its begin stamp to its end stamp) and, for a label
that occurs twice in one segment (the FFN gate and up projections, which run concurrently), the median span from
the first begin to the last end ("pair"). Both local ranks are pooled unless --rank is given.
usage: dispatches.py TRACE [--n 3] [--rank R]
"""
import argparse, gzip, json, statistics, collections

p = argparse.ArgumentParser()
p.add_argument('trace')
p.add_argument('--n', type=int, default=3)
p.add_argument('--rank', type=int)
a = p.parse_args()
opener = gzip.open if a.trace.endswith('.gz') else open
single = collections.defaultdict(list)
pair = collections.defaultdict(list)
steps = 0
with opener(a.trace, 'rt') as f:
    for line in f:
        d = json.loads(line)
        if 'labels' not in d or d['n'] != a.n or (a.rank is not None and d['rank'] != a.rank):
            continue
        steps += 1
        open_ = {}
        seg = collections.defaultdict(list)
        for label, t in zip(d['labels'], d['ns']):
            if label == 'begin':
                seg = collections.defaultdict(list)
                open_ = {}
                continue
            if label == 'end':
                for k, spans in seg.items():
                    if len(spans) > 1:
                        pair[k].append(max(e for _, e in spans) - min(b for b, _ in spans))
                continue
            if label in open_:
                b = open_.pop(label)
                single[label].append(t - b)
                seg[label].append((b, t))
            else:
                open_[label] = t
print(f'{steps} steps with {a.n} columns')
print('| dispatch | count | median µs | pair span µs |')
print('|---|---|---|---|')
for label, v in sorted(single.items(), key=lambda kv: -statistics.median(kv[1])):
    ps = pair.get(label)
    print(f'| {label} | {len(v)} | {statistics.median(v) / 1e3:.1f} | {statistics.median(ps) / 1e3:.1f} |' if ps else f'| {label} | {len(v)} | {statistics.median(v) / 1e3:.1f} | - |')
