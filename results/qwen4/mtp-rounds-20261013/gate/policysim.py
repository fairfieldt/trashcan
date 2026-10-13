#!/usr/bin/env python3
"""The engine's rounds replayed on teacher-forced traces (`qwen4 mtp replay --teacher` along runs of `qwen4 mtp
accept`): the policy (spec::Policy: acceptance decaying 0.94 per draft with a prior worth 2 drafts, sizes measured once
at the start, costs fixed here), the shadow draft of k = 0 rounds, and the handling of unsure drafts:

  main      pooled acceptance, every chosen draft verified
  drop      pooled acceptance (dropped drafts tested when their position is known), unsure drafts not verified
  two-rate  P(draft sure) and P(accepted | sure) estimated apart; expected tokens and cost of k drafts with drops

Greedy traces: a draft is accepted when it is the trunk's token. Sampled traces: accepted with the coupled-draft
probability (sampling.py), drawn with a fixed seed, 20 repetitions.
usage: policysim.py DIR [--t 0] [--draft 5.05] [--verify 54.2,90,107.3,127.8] [--plain 54.0] [--unsure 0.6]
"""
import argparse, glob, json, math, os, random, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sampling'))
from sampling import dist, coupled

ap = argparse.ArgumentParser()
ap.add_argument('dir')
ap.add_argument('--t', default='0')
ap.add_argument('--draft', type=float, default=5.05)
ap.add_argument('--verify', default='54.2,90,107.3,127.8')
ap.add_argument('--plain', type=float, default=54.0)
ap.add_argument('--unsure', type=float, default=0.6)
a = ap.parse_args()
D, V, U = a.draft, [float(x) for x in a.verify.split(',')], a.unsure
T = float(a.t)


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
        row = []
        for j, dc in enumerate(step):
            if T == 0:
                p_acc = 1.0 if tg[i + j + 1][0][1] == dc[0][1] else 0.0
            else:
                p_acc = coupled(dist(tg[i + j + 1], T, 64, 0.95), dist(dc, T, 64, 0.95))
            row.append((conf_of(dc), p_acc))
        steps.append(row)
    traces[name] = steps


class Pooled:
    def __init__(self):
        self.hits, self.tests = 1.6, 2.
        self.started = 3  # sizes measured at the start: k = 3, 2, 1, 0

    def tested(self, hit):
        self.hits = self.hits * 0.94 + hit
        self.tests = self.tests * 0.94 + 1

    def sure(self, s):
        pass

    def choose(self):
        if self.started >= 0:
            k, self.started = self.started, self.started - 1
            return k, True
        p = self.hits / self.tests
        best, rate, tok, q = 0, -1, 0., 1.
        for k in range(4):
            tok += q
            q *= p
            r = tok / (k * D + V[k])
            if r > rate:
                best, rate = k, r
        return best, False


class TwoRate(Pooled):
    def __init__(self):
        super().__init__()
        self.s_hits, self.s_tests = 1.6, 2.  # P(sure)

    def sure(self, s):
        self.s_hits = self.s_hits * 0.94 + s
        self.s_tests = self.s_tests * 0.94 + 1

    def choose(self):
        if self.started >= 0:
            k, self.started = self.started, self.started - 1
            return k, True
        acc, c = self.hits / self.tests, self.s_hits / self.s_tests
        s = c * acc
        best, rate = 0, -1
        for k in range(4):
            tokens = sum(s ** j for j in range(k + 1))
            steps = sum(c ** j for j in range(k))
            verify = sum((c ** m * (1 - c) if m < k else c ** k) * V[m] for m in range(k + 1))
            r = tokens / (steps * D + verify)
            if r > rate:
                best, rate = k, r
        return best, False


def simulate(steps, mode, rng):
    pol = TwoRate() if mode == 'two-rate' else Pooled()
    i, tokens, ms, byk = 0, 0, 0., [0] * 4
    while i < len(steps):
        k, refresh = pol.choose()
        s = steps[i]
        if k == 0:
            # shadow round: the depth-1 draft is tested (and its confidence seen)
            ms += V[0]
            pol.sure(s[0][0] >= U)
            if mode != 'two-rate' or s[0][0] >= U:
                pol.tested(rng.random() < s[0][1])
            tokens += 1
            byk[0] += 1
            i += 1
            continue
        drafted, kept, unsure = 0, 0, False
        while kept < k:
            c, _ = s[kept]
            drafted += 1
            if mode != 'main':
                pol.sure(c >= U)
            if mode != 'main' and c < U and not refresh:
                unsure = True
                break
            kept += 1
        outcomes = [rng.random() < s[j][1] for j in range(3)]
        if kept == 0:
            # an unsure first draft: a plain step with it as the shadow draft
            ms += drafted * D + V[0]
            if mode == 'drop':
                pol.tested(outcomes[0])
            tokens += 1
            byk[0] += 1
            i += 1
            continue
        acc = 0
        while acc < kept and outcomes[acc]:
            acc += 1
        for _ in range(acc):
            pol.tested(True)
        if acc < kept:
            pol.tested(False)
        elif unsure and mode == 'drop':
            pol.tested(outcomes[kept])
        ms += drafted * D + V[kept]
        tokens += acc + 1
        byk[kept] += 1
        i += acc + 1
    return tokens, ms, byk


print(f'T {T}: draft {D} ms, rounds {V} ms, plain {a.plain} ms, unsure below {U}')
for mode in ('main', 'drop', 'two-rate'):
    rates = {}
    for p, st in traces.items():
        reps = 1 if T == 0 else 20
        tot_t = tot_ms = 0
        byk = [0] * 4
        for rep in range(reps):
            t_, m_, b_ = simulate(st, mode, random.Random(rep))
            tot_t, tot_ms = tot_t + t_, tot_ms + m_
            byk = [x + y for x, y in zip(byk, b_)]
        rates[p] = (tot_t / tot_ms * 1000, byk)
    chat = [v[0] for p, v in rates.items() if p != 'reference']
    print(f'{mode:9} chat mean {sum(chat) / len(chat):6.2f} (plain {1000 / a.plain:.2f})  ' +
          '  '.join(f'{p} {v[0]:.2f} {"/".join(map(str, v[1]))}' for p, v in rates.items()))
