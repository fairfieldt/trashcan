#!/usr/bin/env python3
"""Acceptance of greedy MTP drafts from `trashcan qwen4 mtp accept` JSON files.

For every decode step i the file holds the generated token g[i] and the drafts the block made from it
(predicting g[i+1], g[i+2], ...). A speculative round starting at step i with k drafts accepts the longest
matching prefix a and emits a + 1 tokens; greedy speculative output equals plain greedy output, so walking the
steps round by round gives the exact tokens per round. With --vocab V a draft at or above V counts as wrong (a
draft head that ranks only the first V tokens).

usage: accept.py FILE.json... [--vocab V]
"""
import json, sys

args = sys.argv[1:]
vocab = None
if '--vocab' in args:
    i = args.index('--vocab')
    vocab = int(args[i + 1])
    del args[i:i + 2]


def accepted(gen, drafts, i, k):
    a = 0
    while a < k and i + 1 + a < len(gen) and drafts[i][a] == gen[i + 1 + a] and (vocab is None or drafts[i][a] < vocab):
        a += 1
    return a


def stats(gen, drafts):
    n = len(drafts)
    depth = len(drafts[0])
    # conditional acceptance of draft j given drafts < j accepted, over every step (complete windows only)
    cond = []
    for j in range(depth):
        ok = tot = 0
        for i in range(n):
            if i + 1 + j >= len(gen):
                continue
            if accepted(gen, drafts, i, j) == j:
                tot += 1
                ok += accepted(gen, drafts, i, j + 1) == j + 1
        cond.append((ok, tot))
    rounds = {}
    for k in range(1, depth + 1):
        i = r = toks = 0
        while i < n:
            a = accepted(gen, drafts, i, k)
            a = min(a, n - 1 - i)  # the last step has no later tokens
            toks += a + 1
            i += a + 1
            r += 1
        rounds[k] = (toks, r)
    return cond, rounds


tot_cond, tot_rounds = None, None
print('file\tsteps\t' + '\t'.join(f'a{j + 1}' for j in range(3)) + '\t' + '\t'.join(f'tok/round k={k}' for k in (1, 2, 3))
      + '\tbelow 32K\tbelow 64K')
for f in args:
    r = json.load(open(f))
    gen, drafts = r['tokens'], r['drafts']
    cond, rounds = stats(gen, drafts)
    if tot_cond is None:
        tot_cond = [[0, 0] for _ in cond]
        tot_rounds = {k: [0, 0] for k in rounds}
    for t, c in zip(tot_cond, cond):
        t[0] += c[0]
        t[1] += c[1]
    for k, v in rounds.items():
        tot_rounds[k][0] += v[0]
        tot_rounds[k][1] += v[1]
    below = lambda v: sum(1 for t in gen[1:] if t < v) / max(1, len(gen) - 1)
    print(f'{f.split("/")[-1]}\t{len(drafts)}\t' + '\t'.join(f'{a / max(b, 1):.3f}' for a, b in cond) + '\t'
          + '\t'.join(f'{t / r:.2f}' for t, r in rounds.values()) + f'\t{below(32768):.3f}\t{below(65536):.3f}')
print('pooled\t\t' + '\t'.join(f'{a / max(b, 1):.3f}' for a, b in tot_cond) + '\t'
      + '\t'.join(f'{t / r:.2f}' for t, r in tot_rounds.values()))
