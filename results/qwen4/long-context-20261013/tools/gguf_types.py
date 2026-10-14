#!/usr/bin/env python3
"""Print name, ggml type and dims of the tensors of GGUF files whose names match a regex.
usage: gguf_types.py REGEX FILE..."""
import re, struct, sys

TYPES = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K',
         12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 20: 'IQ4_NL', 23: 'IQ4_XS', 30: 'BF16'}
pat = re.compile(sys.argv[1])
for path in sys.argv[2:]:
    f = open(path, 'rb')
    rd = lambda fmt: struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
    assert f.read(4) == b'GGUF'
    ver, = rd('I')
    nt, nkv = rd('QQ')
    def rstr():
        n, = rd('Q')
        return f.read(n).decode('utf-8', 'replace')
    SZ = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    def rval(t):
        if t in SZ:
            return rd(SZ[t])[0]
        if t == 8:
            return rstr()
        if t == 9:
            et, n = rd('IQ')
            for _ in range(n):
                rval(et)
            return None
    for _ in range(nkv):
        rstr()
        t, = rd('I')
        rval(t)
    for _ in range(nt):
        name = rstr()
        nd, = rd('I')
        dims = rd('Q' * nd)
        ty, = rd('I')
        off, = rd('Q')
        if pat.search(name):
            print(f"{name:48s} {TYPES.get(ty, ty)!s:8s} {list(dims)}")
