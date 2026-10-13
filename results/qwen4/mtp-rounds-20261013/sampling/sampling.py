#!/usr/bin/env python3
"""Acceptance of MTP drafts when sampling, from runs sampled on the CPU reference (`qwen4 mtp accept --temperature`)
and the engine's drafter replayed along them with the run's tokens (`qwen4 mtp replay --teacher`).

For every step and draft depth, with p the trunk's sampling distribution (temperature, top-k, top-p as the server's
Sampler::probabilities) and q the drafter's:
  greedy  draft = argmax q, kept when the sample equals it: p(argmax q)        (the engine before)
  coupled draft sampled with the target's noise (Gumbel-max), kept when the sample equals it:
          sum_v 1 / sum_w max(p_w / p_v, q_w / q_v)                            (drafter-invariant)
  ss      standard speculative sampling, accept min(1, p/q), residual on rejection: sum_v min(p_v, q_v)
Tokens per round with k drafts: 1 + a1 + a1 a2 + ..., per step, averaged.

usage: sampling.py DIR [--top-k 64] [--top-p 0.95] [--draft-scale 1.0]
"""
import argparse, glob, json, math, os, re


def dist(cands, t, top_k, top_p):
    """Sampler::probabilities: the first top_k candidates (best first), softmax at t, the top-p prefix, renormalized."""
    if t == 0:
        return {cands[0][1]: 1.0}
    c = cands[:top_k]
    mx = c[0][0]
    w = [math.exp((v - mx) / t) for v, _ in c]
    s = sum(w)
    keep, tot = len(c), 0.0
    for i, x in enumerate(w):
        tot += x / s
        if tot >= top_p:
            keep = i + 1
            break
    w = w[:keep]
    s = sum(w)
    return {c[i][1]: w[i] / s for i in range(keep) if w[i] > 0}


def coupled(p, q):
    """P(argmax(log p + g) == argmax(log q + g)) for shared Gumbel noise g."""
    tot = 0.0
    union = set(p) | set(q)
    for v in set(p) & set(q):
        s = 0.0
        for w in union:
            s += max(p.get(w, 0.0) / p[v], q.get(w, 0.0) / q[v])
        tot += 1.0 / s
    return tot


def ss(p, q):
    return sum(min(pv, q.get(v, 0.0)) for v, pv in p.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--top-k', type=int, default=64)
    ap.add_argument('--top-p', type=float, default=0.95)
    ap.add_argument('--draft-scale', type=float, default=1.0, help='draft temperature = scale x the target temperature')
    ap.add_argument('--csv', help='also write every step and depth: prompt, T, step, depth, greedy, coupled, ss')
    a = ap.parse_args()
    csv = open(a.csv, 'w') if a.csv else None
    if csv:
        csv.write('prompt,t,step,depth,greedy,coupled,ss\n')
    rows = []
    for run in sorted(glob.glob(os.path.join(a.dir, 'run-*-t*.json'))):
        m = re.match(r'run-(\w+)-t([\d.]+)\.json', os.path.basename(run))
        name, t = m.group(1), float(m.group(2))
        rep = os.path.join(a.dir, f'rep-{name}-t{m.group(2)}.json')
        if not os.path.exists(rep):
            continue
        r, d = json.load(open(run)), json.load(open(rep))
        targets, tops = r['targets'], d['draft_tops']
        acc = {k: [[], [], []] for k in ('greedy', 'coupled', 'ss')}
        per_round = {k: [0.0, 0.0, 0.0] for k in acc}
        n = 0
        for i, step in enumerate(tops):
            if len(step) < 3:
                continue
            a_ = {k: [] for k in acc}
            for j, dc in enumerate(step):
                p = dist(targets[i + j + 1], t, a.top_k, a.top_p)
                q = dist(dc, t * a.draft_scale, a.top_k, a.top_p)
                g = p.get(dc[0][1], 0.0)
                vals = {'greedy': g, 'coupled': coupled(p, q) if t > 0 else g, 'ss': ss(p, q) if t > 0 else g}
                for k, v in vals.items():
                    acc[k][j].append(v)
                    a_[k].append(v)
                if csv:
                    csv.write(f"{name},{t},{i},{j + 1},{vals['greedy']:.5f},{vals['coupled']:.5f},{vals['ss']:.5f}\n")
            for k in acc:
                tok, prod = 1.0, 1.0
                for depth in range(3):
                    prod *= a_[k][depth]
                    tok += prod
                    per_round[k][depth] += tok
            n += 1
        row = {'prompt': name, 't': t, 'steps': n}
        for k in acc:
            row[k] = {'a': [sum(x) / len(x) for x in acc[k]], 'tokens_per_round': [v / n for v in per_round[k]]}
        rows.append(row)
    print(f"top-k {a.top_k}, top-p {a.top_p}, draft temperature x{a.draft_scale}")
    print(f"{'prompt':12} {'T':>4} {'method':8} {'a1':>6} {'a2':>6} {'a3':>6} {'k=1':>6} {'k=2':>6} {'k=3':>6}")
    by_t = {}
    for row in rows:
        for k in ('greedy', 'coupled', 'ss'):
            x = row[k]
            print(f"{row['prompt']:12} {row['t']:4} {k:8} " + ' '.join(f'{v:6.3f}' for v in x['a'] + x['tokens_per_round']))
            by_t.setdefault((row['t'], k), []).append(x)
    print('mean over prompts')
    for (t, k), xs in sorted(by_t.items()):
        m = [sum(x['a'][i] for x in xs) / len(xs) for i in range(3)] + \
            [sum(x['tokens_per_round'][i] for x in xs) / len(xs) for i in range(3)]
        print(f"{'mean':12} {t:4} {k:8} " + ' '.join(f'{v:6.3f}' for v in m))


if __name__ == '__main__':
    main()
