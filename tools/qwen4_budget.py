#!/usr/bin/env python3
"""VRAM per D700 for Qwen3.8-Flash-Next on a chain of Mac Pros (two GPUs each), from the model's tensor shapes.

Each GPU holds a contiguous range of layers (balanced by bytes, the last GPU also the LM head, or the head split
over the last machine's two GPUs), the KV/indexer caches for its QSA layers, DeltaNet state and batch scratch.
Bits per weight are the GPU layouts: Q4_0 and IQ4_NL with f16 scales 4.5, IQ4_XS 4.25, MXFP4 4.25, IQ2_XXS 2.0625,
Q2_K 2.625 (down padded 640 -> 768 when noted), Q8_0 8.5, f32 32. Usage: qwen4_budget.py [--context N]
"""
import argparse

D, HC, RANK, FF, E, LAYERS, VOCAB = 2560, 4, 320, 640, 512, 48, 248320
QSA = {il for il in range(LAYERS) if (il + 1) % 4 == 0}
GIB, MIB = 1 << 30, 1 << 20
HEAP = 5888 * MIB          # device-local heap per D700 (measured); the budget reported 5874-5881 MiB
SCRATCH = 160 * MIB        # activations, MoE grouping and attention scratch at batch 256 (see results)


def dense_params(il):
    hc = 2 * (2 * HC * D * RANK + HC * D * HC)            # two mixers: down, up, inject
    shexp = 3 * D * FF
    if il in QSA:
        mixer = D * 2 * 24 * 256 + 2 * D * 512 + 24 * 256 * D  # q|gate, k, v, out
    else:
        mixer = D * 10240 + D * 6144 + 6144 * D                 # qkv, z, out
    return hc, shexp, mixer


def layer_bytes(il, r):
    hc, shexp, mixer = dense_params(il)
    gate_up = 2 * E * D * FF * r['gate_up'] / 8
    down = E * FF * D * r['down'] / 8 * r.get('down_pad', 1.0)
    dense = (hc + mixer) * r['dense'] / 8 + shexp * r['shared'] / 8
    small = (E + 1) * D * 4 + (2 * 48 * D * 4 if il not in QSA else (512 + 128) * D * 4)  # router f32, alpha/beta or indexer
    ple = (D * HC * D + D * D) * r['dense'] / 8 + 9 * HC * D * 4 if il == 1 else 0
    return gate_up + down + dense + small + ple


def state_bytes(il, context):
    if il in QSA:
        return context * (2 * 256 * 2 * 2 + 128 * 4) + context // 4 * 128 * 4   # f16 K/V, f32 raw and pooled keys
    return 48 * 128 * 128 * 4 + 3 * 10240 * 4


def plan(machines, r, context, head_bits):
    gpus = 2 * machines
    head = VOCAB * D * head_bits / 8
    costs = [layer_bytes(il, r) + state_bytes(il, context) for il in range(LAYERS)]
    # Contiguous ranges, greedy against the mean; the head split over the last machine's two GPUs.
    target = (sum(costs) + head) / gpus
    ranges, start, acc = [], 0, 0.0
    for il, c in enumerate(costs):
        left = LAYERS - il
        if (acc + c / 2 > target * (len(ranges) + 1) and len(ranges) < gpus - 1) or left == gpus - len(ranges) - 1:
            if il > start:
                ranges.append((start, il))
                start = il
        acc += c
    ranges.append((start, LAYERS))
    out = []
    for g, (a, b) in enumerate(ranges):
        used = sum(costs[a:b]) + SCRATCH + (head / 2 if g >= gpus - 2 else 0)
        out.append((a, b, used))
    return out


RECIPES = {
    'Q4_0 experts, Q8_0 dense': dict(gate_up=4.5, down=4.5, dense=8.5, shared=8.5),
    'IQ4_XS gate/up + IQ4_NL down, Q8_0 dense': dict(gate_up=4.25, down=4.5, dense=8.5, shared=8.5),
    'MXFP4 experts, Q8_0 dense': dict(gate_up=4.25, down=4.25, dense=8.5, shared=8.5),
    'IQ2_XXS gate/up + IQ4_NL down, Q8_0 dense': dict(gate_up=2.0625, down=4.5, dense=8.5, shared=8.5),
    'IQ2_XXS gate/up + Q2_K down (768), Q8_0 dense': dict(gate_up=2.0625, down=2.625, down_pad=1.2, dense=8.5, shared=8.5),
}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--context', type=int, default=32768)
    ap.add_argument('--head-bits', type=float, default=4.5)
    a = ap.parse_args()
    print(f'device-local heap {HEAP / GIB:.2f} GiB per D700; scratch {SCRATCH / MIB:.0f} MiB; context {a.context}; '
          f'head {a.head_bits} bpw split over the last two GPUs')
    for machines in (6, 4):
        for name, r in RECIPES.items():
            p = plan(machines, r, a.context, a.head_bits)
            worst = max(u for _, _, u in p)
            print(f'{machines} Macs | {name:48s} | max {worst / GIB:5.2f} GiB '
                  f'({"fits" if worst <= HEAP else "over"} by {abs(HEAP - worst) / MIB:6.0f} MiB) | '
                  + ' '.join(f'{x}-{y}' for x, y, _ in p))
