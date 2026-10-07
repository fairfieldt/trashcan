#!/usr/bin/env python3
"""VRAM per D700 for Qwen3.8-Flash-Next on a chain of Mac Pros (two GPUs each), from the model's shapes.

Mirrors what engine/src/qwen4/gpu.rs allocates: per layer the two mixers (down, up, the fused down+write copy for
decode, the write weights in f32), DeltaNet or QSA matrices and state, the f32 router, the stacked experts (shared
expert included, in the routed format), the n-gram module at layer 1, K/V caches (f16) and, past 2051 tokens of
context, the indexer caches (f32); per stage the batch scratch; on the head stages the final mixer and the head
(Q6_K repacked to int8 with f32 scales per 16, 10 bits per weight). `--check` compares the model with stages
measured on incite. Each GPU holds a contiguous range of layers; the head goes to the last `--head-split` GPUs
(only 1 is implemented).

`--lean` models four savings that are not implemented: prefill reuses the fused mixer matrix (no separate down
copy), the router in Q8_0, the indexer projections in Q8_0 and a raw indexer-key ring of batch + 4 tokens.

GPU formats (bits per weight): dense Q4_0 5.0 (f32 scales), Q8_0 8.5; experts Q4_0 and IQ4_NL 4.5, IQ4_XS 4.25
(implemented); MXFP4 4.25, IQ3_XXS 3.0625, IQ2_XXS 2.0625 (no GPU kernels yet).
Usage: qwen4_budget.py [--context N] [--batch N] [--head-split N] [--lean] [--check]
"""
import argparse

D, HC, RANK, FF, E, LAYERS, VOCAB = 2560, 4, 320, 640, 512, 48, 248320
W = HC * D
QSA = {il for il in range(LAYERS) if (il + 1) % 4 == 0}
PLE_LAYER = 1
HEADS, KV_HEADS, HEAD_DIM = 24, 2, 256
V_HEADS, K_HEADS, STATE = 48, 16, 128
CONV_DIM = 2 * K_HEADS * STATE + V_HEADS * STATE   # 10240
VALUE_DIM = V_HEADS * STATE                        # 6144
QD = HEADS * HEAD_DIM                              # 6144
IDX_HEADS, IDX_DIM, KPOOL, BUDGET = 4, 128, 4, 512
MIB = 1 << 20
HEAP = 5888 * MIB      # device-local heap per D700; with a desktop running RADV's budget is 5759-5874 MiB
HEAD_BITS = 10.0       # quant::repack_q6: int8 weights and an f32 scale per 16
Q8 = 8.5


def mixer_bytes(dense, inject=True, lean=False):
    down = 0 if (lean and inject) else RANK * W * dense / 8   # lean: prefill uses the fused matrix too
    up = W * RANK * dense / 8
    fused = (RANK + HC) * W * dense / 8 if inject else 0       # decode: down and write rows in one matrix
    write = HC * W * 4 if inject else 0                         # write weights in f32
    return down + up + fused + write + HC * D * 4


def layer_bytes(il, r, context, batch, lean=False):
    dense = r['dense']
    b = 2 * mixer_bytes(dense, lean=lean)
    sparse = context > BUDGET * KPOOL + KPOOL - 1
    if il in QSA:
        b += (D * 2 * QD + 2 * D * KV_HEADS * HEAD_DIM + QD * D) * dense / 8     # q|gate, k, v, out
        b += 2 * context * KV_HEADS * HEAD_DIM * 2                                # f16 K and V
        if sparse:
            b += (IDX_HEADS * IDX_DIM + IDX_DIM) * D * (Q8 / 8 if lean else 4)    # indexer projections
            b += (batch + KPOOL if lean else context) * IDX_DIM * 4               # raw keys
            b += context // KPOOL * IDX_DIM * 4                                   # pooled keys
    else:
        b += (D * CONV_DIM + D * VALUE_DIM + VALUE_DIM * D) * dense / 8           # qkv, z, out
        b += 2 * V_HEADS * D * 4 + CONV_DIM * 4 * 4 + 3 * CONV_DIM * 4            # alpha/beta f32, conv w, conv state
        b += V_HEADS * STATE * STATE * 4                                          # recurrent state
    b += (E + 1) * D * (Q8 / 8 if lean else 4)                                    # router + shared gate
    b += (E + 1) * 2 * FF * D * r['gate_up'] / 8 + (E + 1) * D * FF * r['down'] / 8
    if il == PLE_LAYER:
        b += (D * W + D * D) * dense / 8 + 9 * W * 4 + 3 * W * 4                  # key, value, history, norms
    return b


def scratch_bytes(batch, context):
    sparse = context > BUDGET * KPOOL + KPOOL - 1
    per_token = (6 * W + RANK + 2 * (RANK + 8) + 4 * D + HC + max(CONV_DIM, 2 * QD) + CONV_DIM + VALUE_DIM
                 + 2 * V_HEADS + max(VALUE_DIM, QD) + 2 * KV_HEADS * HEAD_DIM + QD + (E + 1) + 3 * 11
                 + 11 * FF + (11 * D if batch >= 16 else 0) + IDX_DIM + 2 * IDX_HEADS * IDX_DIM
                 + ((context // KPOOL + BUDGET) if sparse else 0))
    return per_token * batch * 4 + HEADS * 64 * (HEAD_DIM + 2) * 4 * 8


def head_bytes(r):
    return VOCAB * D * HEAD_BITS / 8 + mixer_bytes(r['dense'], inject=False) + VOCAB * 4


def stage_bytes(a, b, r, context, batch, head=0.0, lean=False):
    return sum(layer_bytes(il, r, context, batch, lean) for il in range(a, b)) + scratch_bytes(batch, context) + head


def plan(machines, r, context, batch, split, lean):
    """Contiguous ranges, as even as possible in layers (all layers are about the same size)."""
    gpus = 2 * machines
    base, extra = divmod(LAYERS, gpus)
    ranges, start = [], 0
    for g in range(gpus):
        n = base + (1 if g < extra else 0)
        ranges.append((start, start + n))
        start += n
    share = head_bytes(r) / min(split, gpus)
    return [(a, b, stage_bytes(a, b, r, context, batch, 0, lean), share if g >= gpus - split else 0)
            for g, (a, b) in enumerate(ranges)]


RECIPES = {
    'Q4_0 experts': dict(gate_up=4.5, down=4.5, dense=Q8),
    'IQ4_XS gate/up + IQ4_NL down': dict(gate_up=4.25, down=4.5, dense=Q8),
    'IQ4_XS gate/up + MXFP4 down': dict(gate_up=4.25, down=4.25, dense=Q8),
    'IQ3_XXS gate/up + IQ4_NL down': dict(gate_up=3.0625, down=4.5, dense=Q8),
    'IQ2_XXS gate/up + IQ4_NL down': dict(gate_up=2.0625, down=4.5, dense=Q8),
    'IQ2_XXS gate/up + MXFP4 down': dict(gate_up=2.0625, down=4.25, dense=Q8),
}

# Stages measured on incite (`GPUn: ... MiB allocated`), results/qwen4/gpu-20261006.
G_Q4 = dict(gate_up=4.5, down=4.5, dense=5.0)
G_Q8D = dict(gate_up=4.5, down=4.5, dense=Q8)
MEASURED = [
    ('G-q4_0 0:3, ctx 2048, batch 256', G_Q4, 0, 3, 2048, 256, 4405.5),
    ('G-q4_0 3:6, ctx 2048, batch 256', G_Q4, 3, 6, 2048, 256, 4380.5),
    ('G-q8d 0:3, ctx 64, batch 4', G_Q8D, 0, 3, 64, 4, 4369.9),
    ('G-q8d 3:6, ctx 64, batch 4', G_Q8D, 3, 6, 64, 4, 4324.0),
    ('G-q8d 0:3, ctx 2048, batch 256', G_Q8D, 0, 3, 2048, 256, 4516.1),
    ('G-q8d 3:6, ctx 2048, batch 256', G_Q8D, 3, 6, 2048, 256, 4474.1),
    ('G-q8d 2:4, ctx 32768, batch 256', G_Q8D, 2, 4, 32768, 256, 3128.1),
]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--context', type=int, default=32768)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--head-split', type=int, default=1)
    ap.add_argument('--lean', action='store_true')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    if a.check:
        for name, r, x, y, ctx, batch, mib in MEASURED:
            p = stage_bytes(x, y, r, ctx, batch) / MIB
            print(f'{name:34s} measured {mib:7.1f} MiB, model {p:7.1f} MiB ({p - mib:+.1f})')
        raise SystemExit
    print(f'MiB per D700 (device-local heap {HEAP / MIB:.0f}); dense Q8_0; context {a.context}, batch {a.batch}; '
          f'head {head_bytes(RECIPES["Q4_0 experts"]) / MIB:.0f} MiB over the last {a.head_split} GPU(s)'
          f'{"; lean" if a.lean else ""}')
    for machines in (6, 5, 4):
        for name, r in RECIPES.items():
            p = plan(machines, r, a.context, a.batch, a.head_split, a.lean)
            body = max(u for _, _, u, _ in p)
            worst = max(u + h for _, _, u, h in p)
            print(f'{machines} Macs | {name:30s} | {p[0][1] - p[0][0]}-{p[-1][1] - p[-1][0]} layers | '
                  f'layers only {body / MIB:5.0f} ({(HEAP - body) / MIB:+5.0f}) | '
                  f'with head share {worst / MIB:5.0f} ({(HEAP - worst) / MIB:+5.0f})')
