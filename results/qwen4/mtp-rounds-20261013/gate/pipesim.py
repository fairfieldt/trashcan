#!/usr/bin/env python3
"""Drafting the next round during a verification, replayed on greedy teacher-forced traces with chains of 7 drafts
(rep7-*.json, `qwen4 mtp replay --depth 7 --teacher` along the greedy runs of sampling/study.sh).

Rounds of K drafts. While round r (pending token at step i, drafts predicting tokens i+1..i+K) is verified, the CPU
runs the block from the trunk row before step i with the run's tokens (what the drafts were if all are accepted):
depth K predicts the bonus token, depths K+1..2K are the next round's drafts. They are usable when every draft of
round r is accepted and the bonus token is the predicted one; the next round then verifies them without draft steps.
Otherwise the next round drafts from the trunk's row as now (K draft steps before its verification).

usage: pipesim.py DIR [--k 3] [--draft 5.1] [--verify 49.4,67,83,106]
"""
import argparse, glob, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument('dir')
ap.add_argument('--k', type=int, default=3)
ap.add_argument('--draft', type=float, default=5.1)
ap.add_argument('--verify', default='49.4,67,83,106')
ap.add_argument('--samp', default='/Users/fairfieldt/dev/claude-work/flashnext-20261012/mtp2/samp')
a = ap.parse_args()
K, D, V = a.k, a.draft, [float(x) for x in a.verify.split(',')]

for rep in sorted(glob.glob(os.path.join(a.dir, 'rep7-*-t0.json'))):
    name = re.match(r'rep7-(\w+)-t0', os.path.basename(rep)).group(1)
    tops = json.load(open(rep))['draft_tops']
    tg = json.load(open(os.path.join(a.samp, f'run-{name}-t0.json')))['targets']

    def hit(i, j):
        # depth j (0-based) of the chain anchored at step i predicts the token at step i + j + 1
        return len(tops[i]) > j and tg[i + j + 1][0][1] == tops[i][j][0][1]

    n = len(tops)
    # acceptance by depth, given every earlier depth accepted
    acc = []
    for j in range(7):
        reach = [i for i in range(n) if len(tops[i]) > j and all(hit(i, m) for m in range(j))]
        acc.append(sum(hit(i, j) for i in reach) / max(1, len(reach)))
    res = {}
    for mode in ('now', 'pipelined'):
        i, tokens, ms, pre, used, rounds = 0, 0, 0., None, 0, 0
        while i < n - 2 * K - 1:
            if mode == 'pipelined' and pre is not None:
                # drafts from the chain anchored at pre (depths K+1..2K), no draft steps before the batch
                drafts_hit = [hit(pre, K + 1 + m) for m in range(K)]
                cost = V[K]
                used += 1
            else:
                drafts_hit = [hit(i, m) for m in range(K)]
                cost = K * D + V[K]
            acc_n = 0
            while acc_n < K and drafts_hit[acc_n]:
                acc_n += 1
            # the chain anchored at this round's step i, run during its verification
            pre = i if (acc_n == K and hit(i, K)) else None
            tokens += acc_n + 1
            ms += cost
            rounds += 1
            i += acc_n + 1
        res[mode] = (tokens / ms * 1000, tokens / rounds, used / rounds)
    print(f'{name:12} acceptance by depth ' + ' '.join(f'{x:.2f}' for x in acc) +
          f' | K={K}: now {res["now"][0]:.2f} tok/s ({res["now"][1]:.2f}/round), pipelined {res["pipelined"][0]:.2f}'
          f' ({res["pipelined"][1]:.2f}/round, {res["pipelined"][2]:.0%} rounds pre-drafted), '
          f'{res["pipelined"][0] / res["now"][0] - 1:+.1%}')
