#!/usr/bin/env python3
"""Session 4 six-Mac results (runs/NAME/NAME.json from fetch.sh): the qwen4_matf32_t pairs, the chunk-size sweep, the
base/new benches per prompt size (prefill and decode, each run and the mean ratio), the timeline and 100K runs.
usage: s4sum.py [TAG]"""
import json, os, sys, statistics as st
tag = sys.argv[1] if len(sys.argv) > 1 else 's4'
R = os.path.expanduser('~/dev/claude-work/flashnext-20261012/long/runs')
def get(name):
    try:
        return json.load(open(f"{R}/{name}/{name}.json"))
    except Exception:
        return None
def line(name):
    r = get(name)
    if not r:
        return f"{name:30s} no result"
    last = [round(sum(x), 1) for x in r.get('prefill_last_chunk_layer_ms', [])]
    return (f"{name:30s} pp {r['pp']:6d} {r['pp_tok_s']:7.1f} tok/s ({r['pp_seconds']:6.2f} s)  tg {r['tg_tok_s']:6.2f} tok/s"
            f"  last chunk GPU ms {last}")
def rates(names):
    rs = [get(n) for n in names]
    rs = [r for r in rs if r]
    return [r['pp_tok_s'] for r in rs], [r['tg_tok_s'] for r in rs]
print("== qwen4_matf32_t: 2048 tokens, X (without) and NEW, two alternated pairs")
for n in [f"long-{tag}-pp2048-{b}-{r}" for r in (1, 2) for b in (('x', 'new') if r == 1 else ('new', 'x'))]:
    print(line(n))
px, tx = rates([f"long-{tag}-pp2048-x-{r}" for r in (1, 2)])
pn, tn = rates([f"long-{tag}-pp2048-new-{r}" for r in (1, 2)])
if px and pn:
    print(f"   prefill new/x {st.mean(pn) / st.mean(px):.4f}, decode new/x {st.mean(tn) / st.mean(tx):.4f}")
print("== chunks queued behind the running one: 8192 tokens, S (main's order) and NEW, two alternated pairs")
for n in [f"long-{tag}-pp8192q-{b}-{r}" for r in (1, 2) for b in (('s', 'new') if r == 1 else ('new', 's'))]:
    print(line(n))
ps, ts = rates([f"long-{tag}-pp8192q-s-{r}" for r in (1, 2)])
pn, tn = rates([f"long-{tag}-pp8192q-new-{r}" for r in (1, 2)])
if ps and pn:
    print(f"   prefill new/s {st.mean(pn) / st.mean(ps):.4f}, decode new/s {st.mean(tn) / st.mean(ts):.4f}")
print("== chunk size")
for n in [f"long-{tag}-pp3072-c64", f"long-{tag}-pp3072-c128", f"long-{tag}-pp4096-c64",
          f"long-{tag}-pp4096-c128"]:
    print(line(n))
print("== benches")
for pp in (512, 8192, 16384, 30000):
    names = [f"long-{tag}-pp{pp}-{b}-{r}" for r in (1, 2) for b in ('base', 'new')]
    for n in names:
        if get(n):
            print(line(n))
    pb, tb = rates([f"long-{tag}-pp{pp}-base-{r}" for r in (1, 2)])
    pn, tn = rates([f"long-{tag}-pp{pp}-new-{r}" for r in (1, 2)])
    if pb and pn:
        print(f"   {pp}: prefill {st.mean(pb):.1f} -> {st.mean(pn):.1f} tok/s ({st.mean(pn) / st.mean(pb) - 1:+.1%}), "
              f"decode {st.mean(tb):.2f} -> {st.mean(tn):.2f} tok/s ({st.mean(tn) / st.mean(tb) - 1:+.1%}) "
              f"[{len(pb)} base, {len(pn)} new runs]")
print("== timeline, 100K")
for n in (f"long-{tag}-tl-pp8192-new", f"long-{tag}-pp100k-new"):
    print(line(n))
