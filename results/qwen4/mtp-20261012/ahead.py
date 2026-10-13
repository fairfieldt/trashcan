#!/usr/bin/env python3
"""Drafting ahead during verification, simulated on `qwen4 mtp replay --depth D` runs.

A plain round drafts k tokens (k draft steps on the critical path), verifies them with the pending token and keeps
the matching prefix. Drafting ahead: while the chain verifies, the driver's CPU (otherwise idle) continues the same
chain of drafts. When every draft is accepted and the target's next token equals the next chained draft, the
following round sends the chain's next k drafts at once, with no draft step on the critical path; otherwise it
drafts afresh from the trunk's residual. tok/s = tokens / (fresh rounds * k * draft_ms + rounds * verify_ms(k)).

usage: ahead.py DRAFT_MS VERIFY_MS_K1,K2,K3 FILE.json...
"""
import json, sys

draft_ms = float(sys.argv[1])
verify = [float(v) for v in sys.argv[2].split(',')]
runs = [json.load(open(f)) for f in sys.argv[3:]]


def simulate(gen, drafts, k, ahead):
    depth = len(drafts[0])
    i = rounds = fresh = 0
    chain = None  # (start step, offset) of drafts available without drafting
    while i + 1 < len(gen):
        if chain is not None:
            s, off = chain
            ds = drafts[s][off:off + k]
        else:
            ds = drafts[i][:k]
            fresh += 1
            s, off = i, 0
        a = 0
        while a < len(ds) and i + 1 + a < len(gen) and ds[a] == gen[i + 1 + a]:
            a += 1
        rounds += 1
        nxt = off + a  # index in the chain of the draft for the bonus position
        bonus_ok = a == len(ds) and nxt < depth and i + 1 + a < len(gen) and drafts[s][nxt] == gen[i + 1 + a]
        i += a + 1
        chain = (s, nxt + 1) if ahead and bonus_ok and nxt + 1 + k <= depth else None
    return len(gen) - 1, rounds, fresh


for k in (1, 2, 3):
    for ahead in (False, True):
        t = r = f = 0
        for run in runs:
            a, b, c = simulate(run['tokens'], run['drafts'], k, ahead)
            t, r, f = t + a, r + b, f + c
        ms = f * k * draft_ms + r * verify[k - 1]
        print(f"k={k} {'ahead' if ahead else 'plain'}: {t / r:.2f} tokens/round, fresh rounds {f / r:.2f}, "
              f"{1000 * t / ms:.1f} tok/s")
