#!/usr/bin/env python3
"""Prefill timeline of a six-Mac chain run with TRASHCAN_Q4_TIMELINE=1 TRASHCAN_TP_PROFILE=1 (pipeline.rs `tl`).

Reads the driver's stderr and the five node logs (`tl HOST_US EVENT STAGE POS A B` lines; host CLOCK_MONOTONIC us,
per Mac) and prints, per Mac and GPU stage: chunks, GPU busy time per chunk (from the GPU timestamps), the period
between chunk starts, idle time between chunks and what the host was doing during it (waiting for the chunk to
arrive, collect/submit of the other stage, sending), and per-chunk host costs.

usage: timeline.py DRIVER.err NODE-indite.log NODE-inflight.log ... [--chunks A:B] [--csv OUT]"""
import sys, collections, statistics as st

def load(path):
    ev = []
    for line in open(path, errors='replace'):
        if not line.startswith('tl '):
            continue
        f = line.split()
        if len(f) < 7:
            continue
        ev.append((int(f[1]), f[2], int(f[3]), int(f[4]), float(f[5]), float(f[6])))
    return ev

def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float('nan')

def summarize(name, ev):
    by = collections.defaultdict(dict)  # (event, stage, pos) -> (t, a, b)
    for t, e, s, pos, a, b in ev:
        by[(e, s)].setdefault(pos, (t, a, b))
    out = {}
    stages = sorted({s for (e, s) in by if e == 'done'})
    print(f"== {name}: {len(ev)} events")
    for s in stages:
        done = by[('done', s)]
        sub = by[('sub', s)]
        wait = by[('wait', s)]
        poss = sorted(p for p in done if done[p][1] > 0)
        if len(poss) < 3:
            continue
        busy = [(done[p][2] - done[p][1]) / 1e3 for p in poss]
        starts = [done[p][1] for p in poss]
        ends = [done[p][2] for p in poss]
        period = [(starts[i + 1] - starts[i]) / 1e3 for i in range(len(poss) - 1)]
        idle = [(starts[i + 1] - ends[i]) / 1e3 for i in range(len(poss) - 1)]
        # submit latency: host sub event to GPU start
        lat = [(done[p][1] - sub[p][0]) / 1e3 for p in poss if p in sub]
        # collect: host wait start vs GPU end (positive: host was late)
        late = [(wait[p][0] - done[p][2]) / 1e3 for p in poss if p in wait]
        coll = [(done[p][0] - max(wait[p][0], done[p][2])) / 1e3 for p in poss if p in wait]
        span = (ends[-1] - starts[0]) / 1e6
        print(f"  stage {s}: {len(poss)} chunks over {span:.2f} s, GPU busy {sum(busy)/1e3:.2f} s "
              f"({100*sum(busy)/1e3/span:.0f}%); busy/chunk median {st.median(busy):.1f} ms "
              f"(first {busy[0]:.1f}, last {busy[-1]:.1f}); period median {st.median(period):.1f} p90 {q(period,0.9):.1f}; "
              f"idle median {st.median(idle):.1f} p90 {q(idle,0.9):.1f} sum {sum(idle)/1e3:.2f} s")
        print(f"           submit->GPU start median {st.median(lat):.2f} ms; host late to collect median {st.median(late):.1f} ms "
              f"(p90 {q(late,0.9):.1f}); collect after GPU end median {st.median(coll):.2f} ms")
        out[s] = dict(poss=poss, busy=busy, starts=starts, ends=ends)
    # host events
    if ('recv', 0) in by:
        rw = [v[1] / 1e3 for v in by[('recv', 0)].values()]
        print(f"  recv: {len(rw)} chunks, blocked waiting for the chunk median {st.median(rw):.1f} ms p90 {q(rw,0.9):.1f} sum {sum(rw)/1e3:.2f} s")
    if ('send', 0) in by and ('sent', 0) in by:
        sd = [(by[('sent', 0)][p][0] - by[('send', 0)][p][0]) / 1e3 for p in by[('send', 0)] if p in by[('sent', 0)]]
        print(f"  send: {len(sd)} chunks, median {st.median(sd):.1f} ms p90 {q(sd,0.9):.1f} sum {sum(sd)/1e3:.2f} s")
    if ('drv', 0) in by and ('ple', 0) in by:
        pl = [(by[('ple', 0)][p][0] - by[('drv', 0)][p][0]) / 1e3 for p in by[('drv', 0)] if p in by[('ple', 0)]]
        print(f"  driver embed+ngram: {len(pl)} chunks, median {st.median(pl):.1f} ms p90 {q(pl,0.9):.1f} max {max(pl):.1f} sum {sum(pl)/1e3:.2f} s")
    if ('recv', 0) in by and ('sub', 0) in by:
        rs = [(by[('sub', 0)][p][0] - by[('recv', 0)][p][0]) / 1e3 for p in by[('recv', 0)] if p in by[('sub', 0)]]
        print(f"  recv -> stage 0 submitted: median {st.median(rs):.1f} ms p90 {q(rs,0.9):.1f}")
    return out

paths = [a for a in sys.argv[1:] if not a.startswith('--')]
for p in paths:
    summarize(p.split('/')[-1], load(p))
