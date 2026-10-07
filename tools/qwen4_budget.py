#!/usr/bin/env python3
"""VRAM per D700 for Qwen3.8-Flash-Next on a chain of Mac Pros (two GPUs each), from the model's shapes.

Mirrors what engine/src/qwen4/gpu.rs allocates: per layer the two mixers (down, up, the fused down+write copy for
decode, the write weights in f32), DeltaNet or QSA matrices and state, the f32 router, the stacked experts (shared
expert included, in the routed format), the n-gram module at layer 1, the QSA K/V caches (`--kv`: f16, q8 = int8
with an f16 scale per 32, or k16v8 = f16 K and q8 V) and, past 2051 tokens of context or with a q8 cache, the
indexer projections (f32), a ring of raw indexer keys and the pooled keys; per stage the batch scratch; on the head
stages the final mixer and their share of the head (Q6_K repacked to int8 with f32 scales per 16, 10 bits per
weight). `--check` compares the model with every stage measured on incite.

`--lean` models three savings that are not implemented: prefill reuses the fused mixer matrix (no separate down
copy), and the router and indexer projections in Q8_0. `--q2-down rco` puts one layer per GPU (RCO's least
sensitive: 1, 5, 8, 12, 16, 20, 24, 30, 33, 36, 40, 45) at Q2_0 down projections, routed and shared.

GPU formats (bits per weight): dense Q4_0 5.0 (f32 scales), Q8_0 8.5; experts Q4_0 and IQ4_NL 4.5, IQ4_XS 4.25,
Q2_0 2.25 (down only); MXFP4 4.25, IQ3_XXS 3.0625, IQ2_XXS 2.0625 (no GPU kernels yet).
Usage: qwen4_budget.py [--context N] [--batch N] [--head-split N] [--lean] [--kv f16|q8|k16v8] [--q2-down rco]
       qwen4_budget.py --grid [--context N]      six Macs, the long-context options side by side
       qwen4_budget.py --check
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
KVD = KV_HEADS * HEAD_DIM
IDX_HEADS, IDX_DIM, KPOOL, BUDGET = 4, 128, 4, 512
MIB = 1 << 20
HEAP = 5888 * MIB      # device-local heap per D700; with a desktop running RADV's budget is 5759-5874 MiB
HEAD_BITS = 10.0       # quant::repack_q6: int8 weights and an f32 scale per 16
Q8 = 8.5
Q2 = 2.25
RCO_Q2 = (1, 5, 8, 12, 16, 20, 24, 30, 33, 36, 40, 45)


def mixer_bytes(dense, inject=True, lean=False):
    down = 0 if (lean and inject) else RANK * W * dense / 8   # lean: prefill uses the fused matrix too
    up = W * RANK * dense / 8
    fused = (RANK + HC) * W * dense / 8 if inject else 0       # decode: down and write rows in one matrix
    write = HC * W * 4 if inject else 0                         # write weights in f32
    return down + up + fused + write + HC * D * 4


def kv_row(q8):
    return HEAD_DIM + HEAD_DIM // 16 if q8 else 2 * HEAD_DIM


def indexed(context, kv):
    return kv != 'f16' or context > BUDGET * KPOOL + KPOOL - 1


def ring_rows(batch):
    return -(-(batch + KPOOL) // KPOOL) * KPOOL


def layer_bytes(il, r, context, batch, lean=False, kv='f16', ring=True, q2=False):
    dense = r['dense']
    b = 2 * mixer_bytes(dense, lean=lean)
    if il in QSA:
        b += (D * 2 * QD + 2 * D * KVD + QD * D) * dense / 8                       # q|gate, k, v, out
        b += context * KV_HEADS * (kv_row(kv == 'q8') + kv_row(kv != 'f16'))        # K and V caches
        if indexed(context, kv):
            b += (IDX_HEADS * IDX_DIM + IDX_DIM) * D * (Q8 / 8 if lean else 4)      # indexer projections
            b += (ring_rows(batch) if ring else context) * IDX_DIM * 4              # raw keys
            b += context // KPOOL * IDX_DIM * 4                                     # pooled keys
    else:
        b += (D * CONV_DIM + D * VALUE_DIM + VALUE_DIM * D) * dense / 8             # qkv, z, out
        b += 2 * V_HEADS * D * 4 + CONV_DIM * 4 * 4 + 3 * CONV_DIM * 4              # alpha/beta, conv w, conv state
        b += V_HEADS * STATE * STATE * 4                                            # recurrent state
    b += (E + 1) * D * (Q8 / 8 if lean else 4)                                      # router + shared gate
    b += (E + 1) * 2 * FF * D * r['gate_up'] / 8 + (E + 1) * D * FF * (Q2 if q2 else r['down']) / 8
    if il == PLE_LAYER:
        b += (D * W + D * D) * dense / 8 + 9 * W * 4 + 3 * W * 4                    # key, value, history, norms
    return b


def scratch_bytes(batch, context, kv='f16'):
    per_token = (6 * W + RANK + 2 * (RANK + 8) + 4 * D + HC + max(CONV_DIM, 2 * QD) + CONV_DIM + VALUE_DIM
                 + 2 * V_HEADS + max(VALUE_DIM, QD) + 2 * KVD + QD + (E + 1) + 3 * 11
                 + 11 * FF + (11 * D if batch >= 16 else 0) + IDX_DIM + 2 * IDX_HEADS * IDX_DIM
                 + ((context // KPOOL + BUDGET) if indexed(context, kv) else 0)
                 + (KVD if kv == 'q8' else 0) + (KVD if kv != 'f16' else 0))           # f32 K/V before the q8 store
    return per_token * batch * 4 + HEADS * 64 * (HEAD_DIM + 2) * 4 * 8


def head_bytes(r, rows=VOCAB, ends=True):
    """Head rows (10-bit repack), their logits and candidate scratch; the stage that ends the model adds the final
    mixer."""
    chunks = -(-rows // 1024)
    return (rows * D * HEAD_BITS / 8 + rows * 4 + chunks * 64 * 8
            + (mixer_bytes(r['dense'], inject=False) if ends else 0))


def stage_bytes(a, b, r, context, batch, head=0.0, lean=False, kv='f16', ring=True, q2=()):
    return (sum(layer_bytes(il, r, context, batch, lean, kv, ring, il in q2) for il in range(a, b))
            + scratch_bytes(batch, context, kv) + head)


def plan(machines, r, context, batch, split, lean, kv='f16', q2=()):
    """Contiguous ranges, as even as possible in layers (all layers are about the same size)."""
    gpus = 2 * machines
    base, extra = divmod(LAYERS, gpus)
    ranges, start = [], 0
    for g in range(gpus):
        n = base + (1 if g < extra else 0)
        ranges.append((start, start + n))
        start += n
    share = head_bytes(r) / min(split, gpus)  # the ending stage's final mixer is small
    return [(a, b, stage_bytes(a, b, r, context, batch, 0, lean, kv, True, q2), share if g >= gpus - split else 0)
            for g, (a, b) in enumerate(ranges)]


RECIPES = {
    'Q4_0 experts': dict(gate_up=4.5, down=4.5, dense=Q8),
    'IQ4_XS gate/up + IQ4_NL down': dict(gate_up=4.25, down=4.5, dense=Q8),
    'IQ4_XS gate/up + MXFP4 down': dict(gate_up=4.25, down=4.25, dense=Q8),
    'IQ3_XXS gate/up + IQ4_NL down': dict(gate_up=3.0625, down=4.5, dense=Q8),
    'IQ2_XXS gate/up + IQ4_NL down': dict(gate_up=2.0625, down=4.5, dense=Q8),
    'IQ2_XXS gate/up + MXFP4 down': dict(gate_up=2.0625, down=4.25, dense=Q8),
}

# Stages measured on incite (`GPUn: ... MiB allocated`), results/qwen4/gpu-20261006 and context-20261006. `ring`:
# the raw indexer keys in a ring (from "Add Q8 K/V caches for QSA ..."); before that they spanned the context.
G_Q4 = dict(gate_up=4.5, down=4.5, dense=5.0)
G_Q8D = dict(gate_up=4.5, down=4.5, dense=Q8)
G_IQ4 = RECIPES['IQ4_XS gate/up + IQ4_NL down']
MEASURED = [  # name, recipe, first, last, context, batch, head, kv, ring, q2 layers, MiB
    ('G-q4_0 0:3, ctx 2048, batch 256', G_Q4, 0, 3, 2048, 256, False, 'f16', False, (), 4405.5),
    ('G-q4_0 3:6, ctx 2048, batch 256', G_Q4, 3, 6, 2048, 256, False, 'f16', False, (), 4380.5),
    ('G-q8d 0:3, ctx 64, batch 4', G_Q8D, 0, 3, 64, 4, False, 'f16', False, (), 4369.9),
    ('G-q8d 3:6, ctx 64, batch 4', G_Q8D, 3, 6, 64, 4, False, 'f16', False, (), 4324.0),
    ('G-q8d 0:3, ctx 2048, batch 256', G_Q8D, 0, 3, 2048, 256, False, 'f16', False, (), 4516.1),
    ('G-q8d 3:6, ctx 2048, batch 256', G_Q8D, 3, 6, 2048, 256, False, 'f16', False, (), 4474.1),
    ('G-q8d 2:4, ctx 32768, batch 256', G_Q8D, 2, 4, 32768, 256, False, 'f16', False, (), 3128.1),
    ('G-iq4 0:3, ctx 64, batch 4', G_IQ4, 0, 3, 64, 4, False, 'f16', False, (), 4219.6),
    ('G-iq4 3:6, ctx 64, batch 4', G_IQ4, 3, 6, 64, 4, False, 'f16', False, (), 4173.7),
    ('G-iq4 0:3, ctx 2048, batch 256', G_IQ4, 0, 3, 2048, 256, False, 'f16', False, (), 4365.8),
    ('G-iq4 3:6, ctx 2048, batch 256', G_IQ4, 3, 6, 2048, 256, False, 'f16', False, (), 4323.8),
    ('G-iq4 0:4, ctx 32768, batch 256', G_IQ4, 0, 4, 32768, 256, False, 'f16', False, (), 5843.0),
    ('G-iq4 46:48 + head, ctx 2048, batch 256', G_IQ4, 46, 48, 2048, 256, True, 'f16', False, (), 3698.7),
    ('G-iq4 44:48 + head, ctx 2048, batch 256', G_IQ4, 44, 48, 2048, 256, True, 'f16', False, (), 6479.9),
    ('G-iq4 0:4 + rows 0..21504, ctx 32768, b 128', G_IQ4, 0, 4, 32768, 128, 21504, 'f16', False, (), 5830.4),
    ('G-iq4 3:6, ctx 64, batch 4, kv q8', G_IQ4, 3, 6, 64, 4, False, 'q8', True, (), 4179.9),
    ('S-q2dn 0:3, ctx 64, batch 4', G_IQ4, 0, 3, 64, 4, False, 'f16', True, (1,), 3994.1),
    ('S-q2dn 3:6, ctx 64, batch 4', G_IQ4, 3, 6, 64, 4, False, 'f16', True, (5,), 3948.3),
    ('S-q2dn 0:3, ctx 2048, batch 256', G_IQ4, 0, 3, 2048, 256, False, 'f16', True, (1,), 4140.3),
    ('S-q2dn 3:6, ctx 2048, batch 256', G_IQ4, 3, 6, 2048, 256, False, 'f16', True, (5,), 4098.3),
    ('S-q2dn 0:4 + rows 0..21504, ctx 204800, b 128, q8', G_IQ4, 0, 4, 204800, 128, 21504, 'q8', True, (1,), 5780.0),
]


def check():
    for name, r, x, y, ctx, batch, head, kv, ring, q2, mib in MEASURED:
        hb = head_bytes(r) if head is True else head_bytes(r, head, ends=False) if head else 0
        p = stage_bytes(x, y, r, ctx, batch, hb, False, kv, ring, q2) / MIB
        print(f'{name:50s} measured {mib:7.1f} MiB, model {p:7.1f} MiB ({p - mib:+.1f})')


def grid(context):
    """Six Macs, IQ4_XS/IQ4_NL experts, head over all 12 GPUs, 128-token chunks: the worst GPU's headroom."""
    r = RECIPES['IQ4_XS gate/up + IQ4_NL down']
    print(f'six Macs, IQ4_XS/IQ4_NL experts, Q8_0 dense, head over 12 GPUs, 128-token chunks, context {context}: '
          f'MiB left on the fullest GPU of {HEAP / MIB:.0f}')
    print(f'{"K/V cache":10s} {"Q2_0 down":14s} {"current":>9s} {"--lean":>9s}')
    for kv in ('f16', 'k16v8', 'q8'):
        for q2 in ((), RCO_Q2):
            row = []
            for lean in (False, True):
                p = plan(6, r, context, 128, 12, lean, kv, q2)
                row.append((HEAP - max(u + h for _, _, u, h in p)) / MIB)
            print(f'{kv:10s} {"1 of 4 layers" if q2 else "none":14s} {row[0]:+9.0f} {row[1]:+9.0f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--context', type=int, default=32768)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--head-split', type=int, default=1)
    ap.add_argument('--lean', action='store_true')
    ap.add_argument('--kv', default='f16', choices=('f16', 'q8', 'k16v8'))
    ap.add_argument('--q2-down', default='', choices=('', 'rco'))
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--grid', action='store_true')
    a = ap.parse_args()
    if a.check:
        check()
        raise SystemExit
    if a.grid:
        grid(a.context)
        raise SystemExit
    q2 = RCO_Q2 if a.q2_down == 'rco' else ()
    print(f'MiB per D700 (device-local heap {HEAP / MIB:.0f}); dense Q8_0; context {a.context}, batch {a.batch}; '
          f'{a.kv} K/V; head {head_bytes(RECIPES["Q4_0 experts"]) / MIB:.0f} MiB over the last {a.head_split} GPU(s)'
          f'{"; lean" if a.lean else ""}{"; Q2_0 down on 1 of 4 layers" if q2 else ""}')
    for machines in (6, 5, 4):
        for name, r in RECIPES.items():
            p = plan(machines, r, a.context, a.batch, a.head_split, a.lean, a.kv, q2)
            body = max(u for _, _, u, _ in p)
            worst = max(u + h for _, _, u, h in p)
            print(f'{machines} Macs | {name:30s} | {p[0][1] - p[0][0]}-{p[-1][1] - p[-1][0]} layers | '
                  f'layers only {body / MIB:5.0f} ({(HEAP - body) / MIB:+5.0f}) | '
                  f'with head share {worst / MIB:5.0f} ({(HEAP - worst) / MIB:+5.0f})')
