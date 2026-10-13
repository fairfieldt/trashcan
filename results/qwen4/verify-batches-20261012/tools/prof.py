#!/usr/bin/env python3
"""Stage-bench profile summary: walls, per-stage GPU ms, phase sums over the 8 layers, and per-dispatch microseconds
of one DeltaNet layer (5) and the QSA layer (7) of stage 1, decode (n=1) and the prefill chunk (n=batch).
usage: prof.py [--layers 5,7] [--phases] JSON..."""
import json, sys, collections
args = sys.argv[1:]
layers = [5, 7]
show_disp = True
if args and args[0] == '--layers':
    layers = [int(x) for x in args[1].split(',')]; args = args[2:]
if args and args[0] == '--nodisp':
    show_disp = False; args = args[1:]

def by_layer(disp):
    out, cur = collections.OrderedDict(), None
    for lab, ms, gap in disp:
        if lab.startswith('layer '):
            cur = int(lab.split()[1]); out[cur] = []; continue
        if cur is not None:
            out[cur].append((lab, ms * 1000))
    return out

def phases(stages):
    tot = collections.Counter()
    for s in stages:
        for li, name, ms in s['phases']:
            tot[name] += ms
    return tot

rows = []
for path in args:
    j = json.load(open(path))
    b = j['batch']
    pre = [sum(s['layer_ms']) for s in j['prefill']]
    dec = [sum(s['layer_ms']) for s in j['decode']]
    ph_p, ph_d = phases(j['prefill']), phases(j['decode'])
    print(f"== {path}: batch {b}  chunk wall {j['prefill_wall_ms']:.2f} ms (GPU {sum(pre):.2f}: {' + '.join(f'{x:.2f}' for x in pre)})"
          f"  decode wall {j['decode_wall_ms']:.2f} (GPU {sum(dec):.2f})  chunk/decode wall {j['prefill_wall_ms']/j['decode_wall_ms']:.2f}")
    print("   phases (8 layers, GPU ms) decode | chunk: " + ", ".join(f"{k} {ph_d[k]:.2f}|{ph_p[k]:.2f}" for k in ph_d))
    if not show_disp:
        continue
    D, P = by_layer(j['dispatches'][1]), by_layer(j['prefill_dispatches'][1]) if j['prefill_dispatches'] else {}
    for li in layers:
        if li not in D:
            continue
        print(f"   layer {li}: dispatch | decode us | chunk us")
        for k, (lab, us) in enumerate(D[li]):
            pl = P.get(li, [])
            plab, pus = pl[k] if k < len(pl) else ('', float('nan'))
            print(f"     {lab[:44]:44s} {us:7.1f} | {plab[:40]:40s} {pus:7.1f}")
