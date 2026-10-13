#!/usr/bin/env python3
"""Rounds that stop drafting by the drafter's confidence, simulated on the teacher-forced replays (greedy runs).

A round at step i drafts up to 3 tokens; draft j is accepted when it equals the trunk's token at that position (the
chain is fed the run's tokens, so this is acceptance given the earlier drafts were accepted). Costs: draft step D ms,
verification of n tokens V[n] ms. Rules:
  fixed k        always k drafts
  pooled         the engine's choice: k maximizing (1 + a + .. + a^k) / (k D + V[k+1]) with the run's pooled a
  conf theta     draft while the last draft's confidence (its softmax probability at T = 1) is >= theta, at most 3;
                 the low-confidence draft is still verified
  marginal       continue after draft j while P(drafts 1..j accepted) * abar >= lam * (D + V[j+2] - V[j+1]), where the
                 probabilities come from a confidence calibration and lam is the plain rate (tokens per ms)
usage: confsim.py DIR [--draft 5.1] [--verify 55,91,108,128]
"""
import argparse, glob, json, math, os, re

ap = argparse.ArgumentParser()
ap.add_argument('dir')
ap.add_argument('--draft', type=float, default=5.1)
ap.add_argument('--verify', default='55,91,108,128')
ap.add_argument('--t', default='0')
a = ap.parse_args()
D = a.draft
V = [0.] + [float(x) for x in a.verify.split(',')]


def conf_of(dc):
    mx = dc[0][0]
    return 1 / sum(math.exp(v - mx) for v, _ in dc)


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
        steps.append([(conf_of(dc), tg[i + j + 1][0][1] == dc[0][1]) for j, dc in enumerate(step)])
    traces[name] = steps

# calibration: acceptance by confidence bin, pooled over prompts and depths
edges = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
cnt = [[0, 0] for _ in edges[:-1]]
for steps in traces.values():
    for s in steps:
        for c, ok in s:
            b = max(k for k in range(len(edges) - 1) if c >= edges[k])
            cnt[b][0] += ok
            cnt[b][1] += 1
cal = [x[0] / max(x[1], 1) for x in cnt]


def calib(c):
    return cal[max(k for k in range(len(edges) - 1) if c >= edges[k])]


def simulate(steps, rule):
    """Walk the run: each round starts at the current step; returns tokens per ms and rounds by k."""
    i, tokens, ms, byk = 0, 0, 0., [0] * 4
    n = len(steps)
    while i < n:
        s = steps[i]
        k = rule(s)
        # A rule may draft one more step than it verifies (a draft it drops); charge every draft step computed.
        steps_drafted = getattr(rule, 'drafted', lambda s, k: k)(s, k)
        acc = 0
        while acc < k and s[acc][1]:
            acc += 1
        tokens += acc + 1
        ms += steps_drafted * D + V[k + 1]
        byk[k] += 1
        i += acc + 1
    return tokens / ms * 1000, byk


def fixed(k):
    return lambda s: k


def pooled_rule(steps):
    acc = [ok for s in steps for _, ok in s]
    p = sum(acc) / len(acc)
    best = max(range(4), key=lambda k: sum(p ** j for j in range(k + 1)) / (k * D + V[k + 1]))
    return lambda s: best


def conf_rule(theta):
    def r(s):
        k = 0
        while k < 3:
            k += 1
            if s[k - 1][0] < theta:
                break
        return k
    return r


def marginal_rule(lam, abar):
    def r(s):
        # draft the first only if drafting up to 3 at abar pays
        p, k = 1.0, 0
        while k < 3:
            gain = p * abar
            if gain < lam * (D + V[k + 2] - V[k + 1]) and k > 0:
                break
            if k == 0 and (1 + abar + abar ** 2 + abar ** 3) / (3 * D + V[4]) < lam:
                break
            k += 1
            p *= calib(s[k - 1][0])
        return k
    return r


print('calibration:', ' '.join(f'[{edges[b]},{edges[b + 1]}) {cal[b]:.2f}/{cnt[b][1]}' for b in range(len(cal))))
plain = 1000 / 54.6
print(f'plain {plain:.2f} tok/s; D {D} ms; V {V[1:]}')
rules = {'fixed 1': lambda st: fixed(1), 'fixed 2': lambda st: fixed(2), 'fixed 3': lambda st: fixed(3),
         'pooled': pooled_rule}
for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    rules[f'conf {th}'] = lambda st, th=th: conf_rule(th)
for ab in (0.8, 0.85, 0.9):
    rules[f'marginal abar {ab}'] = lambda st, ab=ab: marginal_rule(1 / 54.6, ab)


def drop_rule(theta):
    """Draft while confident; a draft below theta is not verified (the round keeps the drafts before it), but its
    draft step was computed."""
    def r(s):
        k = 0
        while k < 3 and s[k][0] >= theta:
            k += 1
        return k
    r.drafted = lambda s, k: min(k + 1, 3) if k < 3 else 3
    return r


for th in (0.3, 0.5, 0.6, 0.7, 0.8):
    rules[f'drop {th}'] = lambda st, th=th: drop_rule(th)


def mixed_rule(first, later):
    """An unsure first draft (below `first`) is dropped: the round is a plain step; an unsure later draft (below
    `later`) is verified and ends the drafting."""
    def r(s):
        if s[0][0] < first:
            return 0
        k = 1
        while k < 3:
            k += 1
            if s[k - 1][0] < later:
                break
        return k
    r.drafted = lambda s, k: 1 if k == 0 else k
    return r


for f in (0.5, 0.6, 0.7):
    for l in (0.3, 0.5, 0.6):
        rules[f'mixed {f}/{l}'] = lambda st, f=f, l=l: mixed_rule(f, l)


def pooled_drop(steps, theta):
    """The pooled policy's k as the most drafts, then the drop rule."""
    kmax = pooled_rule(steps)(None)

    def r(s):
        k = 0
        while k < kmax and s[k][0] >= theta:
            k += 1
        return k
    r.drafted = lambda s, k: min(k + 1, kmax) if k < kmax else kmax
    return r


rules['pooled + drop 0.6'] = lambda st: pooled_drop(st, 0.6)


def pooled_confident_drop(steps, theta):
    """The pooled policy's k from the acceptance of confident drafts only, then the drop rule."""
    acc = [ok for s in steps for c, ok in s if c >= theta]
    p = sum(acc) / len(acc)
    kmax = max(range(4), key=lambda k: sum(p ** j for j in range(k + 1)) / (k * D + V[k + 1]))

    def r(s):
        k = 0
        while k < kmax and s[k][0] >= theta:
            k += 1
        return k
    r.drafted = lambda s, k: min(k + 1, kmax) if k < kmax else kmax
    return r


rules['pooled confident + drop 0.6'] = lambda st: pooled_confident_drop(st, 0.6)
for name, mk in rules.items():
    res = {p: simulate(st, mk(st)) for p, st in traces.items()}
    mean = sum(v[0] for v in res.values()) / len(res)
    print(f'{name:20} mean {mean:6.2f} ' + ' '.join(f'{p} {v[0]:.2f} {"/".join(map(str, v[1]))}' for p, v in res.items()))
