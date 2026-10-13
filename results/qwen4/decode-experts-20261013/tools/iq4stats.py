#!/usr/bin/env python3
"""IQ4 code statistics of expert tensors and the shared-memory bank conflicts of candidate decode tables, with the
real data and the gate/up kernel's access pattern (32 consecutive lanes read 32 consecutive nibble words of a row).
usage: iq4stats.py GGUF_SHARD TENSOR_NAME [EXPERTS]"""
import collections, random, struct, sys

path, want = sys.argv[1], sys.argv[2]
nexp = int(sys.argv[3]) if len(sys.argv) > 3 else 4
f = open(path, 'rb')
def rd(fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
assert f.read(4) == b'GGUF'
ver, = rd('I'); nt, nkv = rd('QQ')
def rstr(): n, = rd('Q'); return f.read(n).decode('utf-8', 'replace')
SZ = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
def rval(t):
    if t in SZ: return rd(SZ[t])[0]
    if t == 8: return rstr()
    if t == 9:
        et, n = rd('IQ')
        return [rval(et) for _ in range(n)]
align = 32
for _ in range(nkv):
    k = rstr(); t, = rd('I'); v = rval(t)
    if k == 'general.alignment': align = v
ts = {}
for _ in range(nt):
    name = rstr(); nd, = rd('I'); dims = rd('Q' * nd); ty, off = rd('IQ'); ts[name] = (ty, off, dims)
base = (f.tell() + align - 1) // align * align
ty, off, dims = ts[want]
print(want, 'type', ty, 'dims', dims)
k, rows = dims[0], dims[1]
assert ty == 23 or ty == 20, 'IQ4_XS (30) or IQ4_NL (20)'
nib = collections.Counter(); byt = collections.Counter()
conf = collections.defaultdict(list)
random.seed(1)
for e in range(nexp):
    ex = random.randrange(dims[2]) if len(dims) > 2 else 0
    for r in random.sample(range(rows), 16):
        if ty == 23:
            rb = k // 256 * 136
            f.seek(base + off + (ex * rows + r) * rb)
            raw = f.read(rb)
            words = []
            for sb in range(k // 256):
                q = raw[sb * 136 + 8: sb * 136 + 136]
                words += list(struct.unpack('<32I', q))
        else:
            rb = k // 32 * 18
            f.seek(base + off + (ex * rows + r) * rb)
            raw = f.read(rb)
            words = []
            for b in range(k // 32):
                words += list(struct.unpack('<4I', raw[b * 18 + 2: b * 18 + 18]))
        for w in words:
            for s in range(8):
                nib[(w >> (4 * s)) & 15] += 1
            for s in range(4):
                byt[(w >> (8 * s)) & 255] += 1
        # half-waves of 32 consecutive words; per byte position m: table conflicts
        for h in range(0, len(words) - 31, 32):
            half = words[h:h + 32]
            for m in range(4):
                bs = [(w >> (8 * m)) & 255 for w in half]
                # 256-entry float table at address 4 * byte: bank = byte % 32
                banks = collections.defaultdict(set)
                for x in bs: banks[x % 32].add(x)
                conf['byte256'].append(max(len(v) for v in banks.values()))
                # 16-entry float table (nibble): conflict-free by construction
                for sh in (0, 4):
                    ns = [(x >> sh) & 15 for x in bs]
                    banks = collections.defaultdict(set)
                    for x in ns: banks[x % 32].add(x)
                    conf['nib16'].append(max(len(v) for v in banks.values()))
                # 256-entry float2 table (both nibbles) at address 8 * byte, 16 lanes per pass: bank pair = byte % 16
                for q in range(0, 32, 16):
                    banks = collections.defaultdict(set)
                    for x in bs[q:q + 16]: banks[x % 16].add(x)
                    conf['pair256_b64'].append(max(len(v) for v in banks.values()))
tot = sum(nib.values())
print('nibble histogram (share):', ' '.join(f'{c}:{nib[c] / tot:.3f}' for c in range(16)))
tb = sum(byt.values())
top = byt.most_common(16)
print('distinct bytes', len(byt), 'top 16 bytes cover', f'{sum(v for _, v in top) / tb:.2f}')
for kname, v in conf.items():
    print(f'{kname}: mean conflict degree {sum(v) / len(v):.2f}, max {max(v)}')
