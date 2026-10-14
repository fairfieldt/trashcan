#!/usr/bin/env python3
"""CPU model of the r21 QSA kernels' address arithmetic for one prompt (default: the 3061-token val prompt in the
chain's chunks of 64, context 32768, kpool 4, 512 blocks, max_batch 128), asserting every index stays in its buffer.
The block selection is simulated exactly as qwen4_idx_select.glsl does it (3 radix passes over integer keys, the
group/bin scans, the ballot-ordered output) on adversarial scores (random, all equal, many ties, NaN and +-inf), and
its output must be the exact top-512 with ties to the earlier block, written inside the token's row.
usage: boundscheck.py [TOKENS] [CHUNK] [--decode N]"""
import random, struct, sys

TOK = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3061
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 64
DECODE = int(sys.argv[sys.argv.index('--decode') + 1]) if '--decode' in sys.argv else 10
CTX = int(sys.argv[sys.argv.index("--ctx") + 1]) if "--ctx" in sys.argv else 32768
KIND, M, MAXB, DIM, KVH, HEADS, IDX = 4, 512, 128, 256, 2, 24, 128
STRIDE = CTX // KIND
RING = ((MAXB + KIND + KIND - 1) // KIND) * KIND
SIZE = {  # elements of each buffer, in the unit the kernel indexes
    'pooled_vec4': STRIDE * IDX // 4, 'pooled_f': STRIDE * IDX, 'scores': MAXB * STRIDE, 'sel': MAXB * M,
    'ring': RING * IDX, 'iqr': MAXB * 512, 'krow_uvec4': CTX * KVH * DIM // 8, 'vrow_uvec2': CTX * KVH * DIM // 4,
    'q': MAXB * HEADS * DIM, 'q4': MAXB * HEADS * DIM // 4, 'out': MAXB * HEADS * DIM,
    'partials': HEADS * 64 * (DIM + 2) * 8,
}
worst = {k: -1 for k in SIZE}

def use(buf, i):
    assert 0 <= i < SIZE[buf], f"{buf}[{i}] out of {SIZE[buf]}"
    worst[buf] = max(worst[buf], i)

def key_of(f):
    u = struct.unpack('<I', struct.pack('<f', f))[0]
    return (~u) & 0xffffffff if u & 0x80000000 else u | 0x80000000

def select(keys, nb, wg):
    """qwen4_idx_select for one token: keys[b] for b < nb; returns {slot: block} as the kernel writes them."""
    prefix = pmask = 0
    need = M
    for pas in range(3):
        shift, dmask = (21, 2047) if pas == 0 else ((10, 2047) if pas == 1 else (0, 1023))
        hist = [0] * 2048
        for b in range(nb):
            if keys[b] & pmask == prefix:
                hist[(keys[b] >> shift) & dmask] += 1
        grp = [sum(hist[2047 - l * 32 - i] for i in range(32)) for l in range(64)]
        inc, s_grp, s_acc = 0, None, None
        for lane in range(64):
            own = grp[lane]
            inc += own
            if inc - own < need <= inc:
                assert s_grp is None
                s_grp, s_acc = lane, inc - own
        assert s_grp is not None, "no group holds the rank"
        binc, s_bin, s_gt = 0, None, None
        for lane in range(32):
            b_ = 2047 - s_grp * 32 - lane
            c = hist[b_]
            binc += c
            pre = s_acc + binc - c
            if pre < need <= pre + c:
                assert s_bin is None
                s_bin, s_gt = b_, pre
        assert s_bin is not None, "no bin holds the rank"
        prefix |= s_bin << shift
        pmask |= dmask << shift
        need -= s_gt
        assert need >= 1
    th, ties = prefix, need
    out = {}
    gtbase = eqbase = 0
    j = 0
    while j * wg < nb:
        for w in range(wg // 64):
            pass
        # Lanes in block order: within an iteration, waves in order, lanes in order (ballot exclusive counts).
        gw = ew = 0
        for l in range(wg):
            b = j * wg + l
            g = b < nb and keys[min(b, nb - 1)] > th
            e = b < nb and keys[min(b, nb - 1)] == th
            gb, eb = gtbase + gw, eqbase + ew
            if g or (e and eb < ties):
                slot = gb + min(eb, ties)
                assert 0 <= slot < M, f"slot {slot}"
                assert slot not in out, f"slot {slot} written twice"
                out[slot] = b
            gw += g
            ew += e
        gtbase += gw
        eqbase += ew
        j += 1
    return out

def check_select(scores, nb, wg):
    keys = [key_of(s) for s in scores[:nb]]
    out = select(keys, nb, wg)
    assert sorted(out) == list(range(M)), f"{len(out)} slots written"
    blocks = [out[s] for s in range(M)]
    assert blocks == sorted(blocks), "not in block order"
    want = sorted(range(nb), key=lambda b: (-keys[b], b))[:M]
    assert sorted(want) == blocks, "not the exact top-512 with ties to the earlier block"

def forward(pos, n):
    # qwen4_idx_pool: tokens completing a block.
    for t in range(n):
        p1 = pos + t + 1
        if p1 % KIND:
            continue
        b = p1 // KIND - 1
        for j in range(IDX):
            for i in range(KIND):
                use('ring', ((b * KIND + i) % RING) * IDX + j)
            use('pooled_f', ((j >> 2) * STRIDE + b % STRIDE) * 4 + (j & 3))
    # qwen4_idx_score(_lds): lanes of every group that does not exit.
    tt_max = 8
    for gy in range((n + 7) // 8):
        t0 = gy * 8
        tn = min(tt_max, n - t0)
        nbmax = (pos + t0 + tn) // KIND
        for gx in range((STRIDE + 255) // 256):
            if nbmax <= M or gx * 256 >= nbmax:
                continue
            for l in (0, 255):
                b = gx * 256 + l
                bs = b % STRIDE
                for part in range(2):
                    for j in range(8):
                        use('pooled_vec4', (part * 8 + j) * STRIDE + bs)
                        use('pooled_vec4', (16 + part * 8 + j) * STRIDE + bs)
                for tt in range(tn):
                    for h in range(4):
                        qb = (tt * 4 + h) * 128
                        for x in (qb, qb + 32 + 31 + 64):
                            use('iqr', t0 * 512 + x)
                    nb = (pos + t0 + tt + 1) // KIND
                    if b < nb and nb > M:
                        use('scores', (t0 + tt) * STRIDE + b)
    # qwen4_idx_select: rows of each token.
    for t in range(n):
        nb = (pos + t + 1) // KIND
        if nb <= M:
            for j in range(nb):
                use('sel', t * M + j)
        else:
            use('scores', t * STRIDE + nb - 1)
            use('sel', t * M + M - 1)
    # Attention: prefill (n > 8, grouped) or decode (grouped_dec + merge).
    parts = 0 if n > 8 else (8 if pos + n <= 32 else 16 if pos + n <= 256 else 32 if pos + n <= 1024 else 64)
    for t in range(n):
        p = pos + t
        nb = (p + 1) // KIND
        nsel = min(nb, M)
        selpos = nsel * KIND
        total = selpos + p + 1 - nb * KIND
        assert 1 <= total <= 2051, total
        for i in (0, selpos - 1, selpos, total - 1):
            if 0 <= i < total:
                if i < selpos:
                    use('sel', t * M + i // KIND)
                else:
                    assert nb * KIND + (i - selpos) <= p
        for kh in range(KVH):
            for key in (0, CTX - 1):
                use('krow_uvec4', (key * KVH + kh) * (DIM // 8) + DIM // 8 - 1)
                use('vrow_uvec2', (key * KVH + kh) * (DIM // 4) + 63)
            q0 = (t * HEADS + kh * 12) * DIM
            use('q', q0 + 11 * DIM + DIM - 1)
            use('q4', (t * HEADS + kh * 12 + 11) * 64 + 3 * 16 + 15)
            use('out', (t * HEADS + kh * 12 + 11) * DIM + DIM - 1)
        if parts:
            used = min(parts, (total + 63) // 64)
            assert used * 64 >= total, (parts, total)
            for c in range(parts):
                if c * 64 >= total:
                    continue
                for h in range(HEADS):
                    bse = ((t * HEADS + h) * parts + c) * (DIM + 2)
                    use('partials', bse + 2 + 4 * 63 + 3)
            for h in range(HEADS):
                use('partials', (t * HEADS + h) * parts * (DIM + 2) + (used - 1) * (DIM + 2) + 2 + 255)

random.seed(1)
pos = 0
chunks = []
while pos < TOK:
    n = min(CHUNK, TOK - pos)
    chunks.append((pos, n))
    pos += n
chunks += [(TOK + i, 1) for i in range(DECODE)]
for pos, n in chunks:
    assert pos + n <= CTX and n <= MAXB
    forward(pos, n)
# The selection on adversarial scores at every distinct block count past 512 that the prompt reaches.
nbs = sorted({(p + t + 1) // KIND for p, n in chunks for t in range(n) if (p + t + 1) // KIND > M})
print(f"{len(chunks)} chunks; block counts past {M}: {nbs[0]}..{nbs[-1]} ({len(nbs)})")
kinds = {
    'random': lambda nb: [random.gauss(0, 3) for _ in range(nb)],
    'all equal': lambda nb: [1.5] * nb,
    'few values': lambda nb: [float(random.randint(0, 3)) for _ in range(nb)],
    'nan, inf, zeros': lambda nb: [random.choice([float('nan'), float('inf'), float('-inf'), 0.0, -0.0, 2.0])
                                   for _ in range(nb)],
    'denormals': lambda nb: [random.choice([1e-45, 2e-45, 0.0, 1e-44]) for _ in range(nb)],
}
for name, gen in kinds.items():
    for nb in nbs[:: max(1, len(nbs) // 12)] + [nbs[-1]]:
        s = gen(nb)
        for wg in (256, 1024):
            check_select(s, nb, wg)
    print(f"selection exact and in bounds: {name}")
for buf, i in worst.items():
    print(f"{buf:12s} max index {i:>10d} of {SIZE[buf]:>10d}")
print("ok")
