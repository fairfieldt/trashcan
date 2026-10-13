#!/usr/bin/env python3
"""Per-dispatch microseconds of the MoE phase (router, top-k, experts) from stage-bench JSON written with
TRASHCAN_Q4_PROFILE=1: mean over the DeltaNet layers 5 and 6 and the QSA layer 7 of the second stage (GPU 1), for the
decode steps (one token) and the profiled chunk (the --batch tokens at position 0). Also the stage's layer GPU ms.
usage: prof.py RUN.json [RUN.json...]"""
import json, sys, collections

KEYS = ('matf32 513', 'topk', 'moe_', 'router', 'expert')


def layers(disp):
    out, cur = {}, None
    for lab, ms, gap in disp:
        if lab.startswith('layer '):
            cur = int(lab.split()[1]); out[cur] = []
            continue
        if cur is not None:
            out[cur].append((lab, ms * 1000, gap * 1000 if gap < 1e6 else float('nan')))
    return out


def moe_rows(disp, ids=(5, 6, 7)):
    L = layers(disp)
    rows = collections.OrderedDict()
    for li in ids:
        if li not in L:
            continue
        # The MoE dispatches follow the second mixer: from the router (f32 513 rows) to the end of the layer.
        seq = L[li]
        start = next((i for i, (lab, _, _) in enumerate(seq) if ' 513x' in lab), None)
        if start is None:
            continue
        for k, (lab, us, gap) in enumerate(seq[start:]):
            rows.setdefault((k, lab), []).append((us, gap))
    return [(lab, sum(u for u, _ in v) / len(v), sum(g for _, g in v) / len(v)) for (k, lab), v in rows.items()]


if __name__ == '__main__':
    for path in sys.argv[1:]:
        j = json.load(open(path))
        print(f'== {path}')
        for name, key, ph in (('decode (1 token at 64)', 'dispatches', 'decode'),
                              (f'chunk of {j["batch"]} at 0', 'prefill_dispatches', 'prefill')):
            d = j.get(key) or []
            if len(d) < 2:
                continue
            rows = moe_rows(d[1])
            tot = sum(us for _, us, _ in rows)
            print(f'  {name}: MoE dispatches {tot:.1f} us (+ gaps {sum(g for _, _, g in rows):.1f})')
            for lab, us, gap in rows:
                print(f'    {lab:40s} {us:7.1f} us  gap {gap:4.1f}')
            lm = j[ph][1]['layer_ms'] if ph in j else []
            ph_ms = collections.defaultdict(float)
            for li, name2, ms in j[ph][1]['phases']:
                ph_ms[name2] += ms
            print(f'    stage 1 layer ms {[round(x, 3) for x in lm]}; phases ' +
                  ', '.join(f'{k} {v:.3f}' for k, v in ph_ms.items()))
        print(f'  decode wall {j["decode_wall_ms"]:.3f} ms, prefill wall {j["prefill_wall_ms"]:.3f} ms')
