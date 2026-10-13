#!/usr/bin/env python3
"""confsim.py for sampled runs: each draft's acceptance is the coupled-draft probability (sampling.py) realized by
Bernoulli draws (20 repetitions); the confidence gate reads the drafter's top probability at T = 1 or at the sampling
temperature. usage: confsim2.py DIR T [--draft 5.1] [--verify 55,91,108,128]"""
import argparse, glob, json, math, os, random, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sampling'))
from sampling import dist, coupled

ap = argparse.ArgumentParser()
ap.add_argument('dir')
ap.add_argument('t')
ap.add_argument('--draft', type=float, default=5.1)
ap.add_argument('--verify', default='55,91,108,128')
a = ap.parse_args()
D = a.draft
V = [0.] + [float(x) for x in a.verify.split(',')]
T = float(a.t)

traces = {}
for run in sorted(glob.glob(os.path.join(a.dir, f'run-*-t{a.t}.json'))):
    name = re.match(r'run-(\w+)-t', os.path.basename(run)).group(1)
    r = json.load(open(run))
    d = json.load(open(os.path.join(a.dir, f'rep-{name}-t{a.t}.json')))
    tg, tops = r['targets'], d['draft_tops']
    steps = []
    for i, step in enumerate(tops):
        if len(step) < 3:
            break
        row = []
        for j, dc in enumerate(step):
            p, q1, qt = dist(tg[i + j + 1], T, 64, 0.95), dist(dc, 1.0, 64, 1.0), dist(dc, T, 64, 0.95)
            row.append((max(q1.values()), max(qt.values()), coupled(p, qt)))
        steps.append(row)
    traces[name] = steps


def simulate(steps, rule, reps=20):
    tot_t, tot_ms = 0, 0.
    rng = random.Random(1)
    for _ in range(reps):
        i, n = 0, len(steps)
        while i < n:
            s = steps[i]
            k = rule(s)
            acc = 0
            while acc < k and rng.random() < s[acc][2]:
                acc += 1
            tot_t += acc + 1
            tot_ms += getattr(rule, 'drafted', lambda s, k: k)(s, k) * D + V[k + 1]
            i += acc + 1
    return tot_t / tot_ms * 1000


def drop(theta, which):
    """Draft while confident; a draft below theta is not verified, but its draft step was computed."""
    def r(s):
        k = 0
        while k < 3 and s[k][which] >= theta:
            k += 1
        return k
    r.drafted = lambda s, k: min(k + 1, 3)
    return r


def gate(theta, which):
    def r(s):
        k = 0
        while k < 3:
            k += 1
            if s[k - 1][which] < theta:
                break
        return k
    return r


print(f'T {T}: D {D} V {V[1:]}')
for name, rule in [('fixed 2', lambda s: 2), ('fixed 3', lambda s: 3)] + \
        [(f'gate T=1 {th}', gate(th, 0)) for th in (0.4, 0.5, 0.6)] + \
        [(f'gate T={T} {th}', gate(th, 1)) for th in (0.4, 0.5, 0.6)] + \
        [(f'drop T=1 {th}', drop(th, 0)) for th in (0.5, 0.6, 0.7)]:
    res = {p: simulate(st, rule) for p, st in traces.items()}
    print(f'{name:18} mean {sum(res.values()) / len(res):6.2f} ' + ' '.join(f'{p} {v:.2f}' for p, v in res.items()))
