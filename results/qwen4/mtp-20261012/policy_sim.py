#!/usr/bin/env python3
"""Simulate the engine's adaptive rounds (`spec::Policy`) on greedy acceptance traces (`qwen4 mtp accept`: the
generated ids and, at every step, three chained drafts) with measured costs; compare acceptance estimators (one
pooled rate or one per depth), memories and priors with fixed rounds and plain decoding.

`--gate RUN.json` adds the reference prompt as decoded on the chain (`qwen4 mtp spec`'s plain ids): its first 9
drafts right (as on the CPU reference, which stops at the end of the reply), none after (the trunk then repeats one
token the block never drafts; session 3: 1 of 148 shadow drafts right).

usage: policy_sim.py [--gate RUN.json] TRACE.json...
"""
import json, sys

# Session 3 (six Macs, main b5b79c7): ms per verification batch of 1-4 tokens (with rollback), per draft step, per
# plain token.
VERIFY = [68.0, 101.0, 112.5, 134.5]
DRAFT = 6.5
PLAIN = 67.2


def run(trace, kmax, decay=None, prior=(0.8, 4.0), fixed=None, depth=False):
    toks, drafts = trace['tokens'], trace['drafts']
    # Per depth (or one pooled rate in slot 0): decayed hits and tests.
    est = [[prior[0] * prior[1], prior[1]] for _ in range(kmax)]
    i, ms, n = 0, 0.0, 0

    def rate(j):
        e = est[j if depth else 0]
        return e[0] / e[1]

    def tested(j, hit):
        e = est[j if depth else 0]
        e[0], e[1] = e[0] * decay + hit, e[1] * decay + 1

    while i + 1 < len(toks) and i < len(drafts):
        if fixed is not None:
            k = fixed
        else:
            best, r, t, p = 0, -1, 0.0, 1.0
            for kk in range(kmax + 1):
                t += p
                if kk < kmax:
                    p *= rate(kk)
                x = t / (kk * DRAFT + VERIFY[kk])
                if x > r:
                    best, r = kk, x
            k = best
        k = min(k, len(toks) - 1 - i)
        d = drafts[i]
        if k == 0:
            if decay:
                tested(0, d[0] == toks[i + 1])
            ms += VERIFY[0]
            i += 1
            n += 1
            continue
        a = 0
        while a < k and d[a] == toks[i + 1 + a]:
            a += 1
        if decay:
            for j in range(min(a + 1, k)):
                tested(j, j < a)
        ms += k * DRAFT + VERIFY[k]
        i += a + 1
        n += a + 1
    return 1000 * n / ms


args = sys.argv[1:]
traces = {}
if args[:1] == ['--gate']:
    ids = json.loads(open(args[1]).readline())['plain_ids']
    traces['gate'] = {'tokens': ids, 'drafts': [ids[i + 1:i + 4] if i < 9 else [-1] * 3 for i in range(len(ids) - 1)]}
    args = args[2:]
traces.update({f.split('/')[-1].removesuffix('.json'): json.load(open(f)) for f in args})
variants = [('plain', None), ('fixed k=1', dict(fixed=1)), ('fixed k=2', dict(fixed=2)), ('fixed k=3', dict(fixed=3))]
for depth in (False, True):
    for decay in (0.94, 0.9, 0.8):
        for w in (4.0, 2.0):
            variants.append((f"{'per depth' if depth else 'pooled'} {decay} 0.8x{w:g}",
                             dict(decay=decay, prior=(0.8, w), depth=depth)))
print('variant\t\t\t' + '\t'.join(traces) + '\tmean of text')
for name, kw in variants:
    rates = [1000 / PLAIN] * len(traces) if kw is None else [run(t, 3, **kw) for t in traces.values()]
    text = [r for n, r in zip(traces, rates) if n != 'gate']
    print(f'{name:24s}\t' + '\t'.join(f'{r:.2f}' for r in rates) + f'\t{sum(text) / len(text):.2f}')
