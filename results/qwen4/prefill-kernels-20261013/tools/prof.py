#!/usr/bin/env python3
"""Per-dispatch prefill profile of stage-bench runs (TRASHCAN_Q4_PROFILE=1, `prefill_dispatches`), layers 0-7 on two
GPUs. Deferred groups (dispatches without a barrier between them) are timed as one span. Prints ms per category summed
over the 8 layers, and the share of the D700's FMA peak (850 MHz x 2048 lanes) for the matrix work.
usage: prof.py RUN.json [RUN.json ...]   (-v: every dispatch of layers 0, 1 and 3)"""
import json, sys, collections

import struct
TICK = struct.unpack('f', struct.pack('f', 1000 / 27))[0] / 1e6  # ms per GPU timestamp tick (27 MHz, f32 period)
WRAP = 2 ** 64 * TICK
PEAK = 850e6 * 2048 / 1e3  # FMAs per ms
D, W, FF, SLOTS = 2560, 10240, 640, 11


def timeline(disp):
    """[(layer, label, start, end)] with absolute times; layer None before the first marker."""
    # A member of a deferred group starts (top of pipe) right after its predecessor started; its end (bottom of pipe)
    # waits for the whole group. The wrapped negative gap is too coarse in f64 to use, so take the predecessor's start.
    out, t, layer, prev_s = [], 0.0, None, 0.0
    for lab, ms, gap in disp:
        if lab.startswith('layer '):
            layer = int(lab.split()[1])
            t += gap
            continue
        s = prev_s if gap > WRAP / 2 else t + gap
        e = s + ms
        out.append((layer, lab, s, e))
        t, prev_s = e, s
    return out


def groups(tl):
    """Merge dispatches that overlap their successor (deferred groups) into spans."""
    res, i = [], 0
    while i < len(tl):
        layer, lab, s, e = tl[i]
        labs = [lab]
        j = i
        while j + 1 < len(tl) and tl[j + 1][2] < e - 1e-6 and tl[j + 1][0] == layer:
            j += 1
            labs.append(tl[j][1])
            e = max(e, tl[j][3])
        res.append((layer, labs, s, e))
        i = j + 1
    return res


def category(layer, labs):
    lab = labs[0]
    name = lab.split()[0]
    shape = lab.split()[1] if ' ' in lab else ''
    gemm = name.startswith('qwen4_q8_gemm')
    if len(labs) > 1:
        shapes = '+'.join(l.split()[1] for l in labs)
        if '12288x2560' in shapes:
            return 'qsa q/k/v proj'
        if '10240x2560' in shapes and '6144x2560' in shapes:
            return 'gdn qkv/z/ab proj'
        if '10240x2560' in shapes and '2560x2560' in shapes:
            return 'n-gram'
        if name.startswith('qwen4_moe_gu'):
            return 'experts gate/up'
        if name == 'qwen_qk':
            return 'qsa qk norm'
        return 'group ' + shapes
    if name in ('qwen4_hc_norm', 'qwen4_mix_act', 'qwen4_q8_gemm_mix', 'qwen4_q8_gemm_s4r4om') or (gemm and shape.startswith('324x')):
        return 'mixers'
    if (gemm and shape == '2560x6144') or name == 'qwen4_sum_parts':
        return 'qsa out proj' if layer in (3, 7) else 'gdn out proj'
    if name in ('qwen4_combine', 'qwen4_ple_sums', 'qwen4_ple_gate', 'qwen4_ple_conv'):
        return 'n-gram'
    m = {'qwen_conv': 'gdn conv', 'qwen_gdn': 'gdn recurrence', 'qwen4_gdn': 'gdn recurrence', 'qwen4_gdn_prep': 'gdn recurrence',
         'qwen4_gdn_rows': 'gdn recurrence', 'qwen4_gated_norm': 'gdn gated norm',
         'qwen_matf32': 'router' if shape.startswith('513x') else 'indexer f32 proj', 'qwen4_topk': 'top-k',
         'qwen4_moe_tiles': 'moe tiles', 'qwen4_moe_sum': 'moe sum', 'qwen_qk': 'qsa qk norm', 'qwen_gate': 'qsa gate',
         'qwen4_kv_store': 'qsa kv store'}
    if name in m:
        return m[name]
    if name.startswith('qwen4_moe_gu'):
        return 'experts gate/up'
    if name.startswith('qwen4_moe_down'):
        return 'experts down'
    if name.startswith('qwen4_idx') or name.startswith('qwen4_attention') or name.startswith('attention') or name.startswith('attn'):
        return 'qsa index+attention'
    if name == 'copy':
        return 'copies'
    return name + ' ' + shape


def flops(cat, n, layers):
    """Multiply-adds of a category over the 8 layers (6 DeltaNet incl. layer 1, 2 QSA)."""
    nd, nq = 6, 2
    return {
        'mixers': 16 * n * (324 * W + W * 320),
        'gdn qkv/z/ab proj': nd * n * (10240 + 6144 + 96) * D,
        'gdn out proj': nd * n * D * 6144,
        'qsa q/k/v proj': nq * n * (12288 + 1024) * D,
        'qsa out proj': nq * n * D * 6144,
        'experts gate/up': 8 * n * SLOTS * 2 * FF * D,
        'experts down': 8 * n * SLOTS * D * FF,
        'router': 8 * n * 513 * D,
        'n-gram': n * (W * D + D * D),
        'experts': 8 * n * SLOTS * 3 * FF * D,
    }.get(cat)


for path in [a for a in sys.argv[1:] if not a.startswith('-')]:
    j = json.load(open(path))
    n = j['batch']
    cats = collections.OrderedDict()
    for stage in j['prefill_dispatches']:
        for layer, labs, s, e in groups(timeline(stage)):
            if layer is None:
                continue
            c = category(layer, labs)
            cats[c] = cats.get(c, 0) + (e - s)
            if '-v' in sys.argv and layer in (0, 1, 3):
                print(f"   L{layer} {'+'.join(labs):70s} {e - s:8.3f}")
    total = sum(cats.values())
    lm = [x for s in j['prefill'] for x in s['layer_ms']]
    print(f"== {path.split('/')[-1]}: batch {n}, wall {j['prefill_wall_ms']:.1f} ms, layers {sum(lm):.1f} ms "
          f"(profiled chunk {total:.1f})")
    for c, ms in sorted(cats.items(), key=lambda x: -x[1]):
        f = flops(c, n, 8)
        eff = f"{f / PEAK / ms * 100:5.1f}% of FMA peak (floor {f / PEAK:6.2f} ms)" if f else ''
        print(f"   {c:22s} {ms:8.2f} ms {ms / total * 100:5.1f}%  {eff}")
