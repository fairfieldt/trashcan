#!/usr/bin/env python3
"""policysim.py's engine policy (two rates, unsure below a fixed threshold dropped) against cost-aware gates on the
same teacher-forced traces:

  fixed     the engine: two-rate choice of k, a draft below --unsure ends the drafting (not verified)
  marginal  two-rate choice of k as the most drafts; draft j is kept while P * g(c_j) >= R * (V(j) - V(j - 1)),
            P the product of g over the round's kept drafts, R the running tokens per ms, g(c) = c * kappa
            (kappa the running ratio of accepted drafts to their summed confidence)
  greedy3   up to 3 drafts every round, the marginal gate, drafting stops when P * a < R * (V(j+1) - V(j) + D)

usage: policysim2.py DIR [--t 0] [--draft 5.1] [--verify 49.4,67,83,106] [--plain 49.2] [--unsure 0.6]
"""
import argparse, glob, json, math, os, random, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sampling'))
from sampling import dist, coupled

ap = argparse.ArgumentParser()
ap.add_argument('dir')
ap.add_argument('--t', default='0')
ap.add_argument('--draft', type=float, default=5.1)
ap.add_argument('--verify', default='49.4,67,83,106')
ap.add_argument('--plain', type=float, default=49.2)
ap.add_argument('--unsure', type=float, default=0.6)
ap.add_argument('--calib', default='scale')  # scale | raw
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


class TwoRate:
    def __init__(self):
        self.hits, self.tests = 1.6, 2.
        self.s_hits, self.s_tests = 1.6, 2.
        self.started = 3
        self.k_hits, self.k_conf = 1.6, 1.8  # calibration: accepted / summed confidence
        self.r_tok, self.r_ms = 1., V[0]  # running rate

    def tested(self, hit, c=None):
        self.hits = self.hits * 0.94 + hit
        self.tests = self.tests * 0.94 + 1
        if c is not None:
            self.k_hits = self.k_hits * 0.94 + hit
            self.k_conf = self.k_conf * 0.94 + c

    def sure(self, s):
        self.s_hits = self.s_hits * 0.94 + s
        self.s_tests = self.s_tests * 0.94 + 1

    def rate(self):
        return self.r_tok / self.r_ms

    def done(self, tokens, ms):
        self.r_tok = self.r_tok * 0.9 + tokens
        self.r_ms = self.r_ms * 0.9 + ms

    def g(self, c):
        if a.calib == 'raw':
            return c
        return min(1., c * self.k_hits / self.k_conf)

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
    pol = TwoRate()
    i, tokens, ms, byk, dropped = 0, 0, 0., [0] * 4, 0
    while i < len(steps):
        k, refresh = pol.choose()
        if mode == 'greedy3' and not refresh:
            k = 3
        s = steps[i]
        if k == 0:
            ms += V[0]
            c0 = s[0][0]
            pol.sure(c0 >= U)
            hit = rng.random() < s[0][1]
            if mode == 'fixed':
                if c0 >= U:
                    pol.tested(hit)
            else:
                pol.tested(hit, c0)
            tokens += 1
            byk[0] += 1
            pol.done(1, V[0])
            i += 1
            continue
        drafted, kept, unsure, P = 0, 0, False, 1.
        R = pol.rate()
        conf_kept = []
        while kept < k:
            c, _ = s[kept]
            drafted += 1
            pol.sure(c >= U)
            if not refresh:
                if mode == 'fixed':
                    drop = c < U
                else:
                    drop = P * pol.g(c) < R * (V[kept + 1] - V[kept])
                if drop:
                    unsure = True
                    break
            P *= pol.g(c)
            conf_kept.append(c)
            kept += 1
            if mode == 'greedy3' and not refresh and kept < 3:
                acc = pol.hits / pol.tests
                if P * acc < R * (V[kept + 1] - V[kept] + D):
                    break
        outcomes = [rng.random() < s[j][1] for j in range(3)]
        if kept == 0:
            ms += drafted * D + V[0]
            if mode != 'fixed':
                pol.tested(outcomes[0], s[0][0])
            tokens += 1
            byk[0] += 1
            dropped += 1
            pol.done(1, drafted * D + V[0])
            i += 1
            continue
        acc_n = 0
        while acc_n < kept and outcomes[acc_n]:
            acc_n += 1
        for j in range(acc_n):
            pol.tested(True, conf_kept[j])
        if acc_n < kept:
            pol.tested(False, conf_kept[acc_n])
        elif unsure and mode != 'fixed':
            pol.tested(outcomes[kept], s[kept][0])
        dropped += unsure
        round_ms = drafted * D + V[kept]
        ms += round_ms
        tokens += acc_n + 1
        byk[kept] += 1
        pol.done(acc_n + 1, round_ms)
        i += acc_n + 1
    return tokens, ms, byk, dropped


print(f'T {T}: draft {D} ms, rounds {V} ms, plain {a.plain} ms, unsure below {U}, calibration {a.calib}')
for mode in ('fixed', 'marginal', 'greedy3'):
    rates = {}
    for p, st in traces.items():
        reps = 1 if T == 0 else 20
        tot_t = tot_ms = 0
        byk = [0] * 4
        drops = 0
        for rep in range(reps):
            t_, m_, b_, d_ = simulate(st, mode, random.Random(rep))
            tot_t, tot_ms = tot_t + t_, tot_ms + m_
            byk = [x + y for x, y in zip(byk, b_)]
            drops += d_
        rates[p] = (tot_t / tot_ms * 1000, byk, drops)
    chat = [v[0] for p, v in rates.items() if p != 'reference']
    print(f'{mode:9} chat mean {sum(chat) / len(chat):6.2f} (plain {1000 / a.plain:.2f})  ' +
          '  '.join(f'{p} {v[0]:.2f} {"/".join(map(str, v[1]))} d{v[2]}' for p, v in rates.items()))
