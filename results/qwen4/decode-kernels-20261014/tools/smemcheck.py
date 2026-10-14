#!/usr/bin/env python3
"""Scalar (SMEM) data loads by buffer binding in ACO programs (isa.py output): for every s_buffer_load, the descriptor
it uses is traced to its s_load_dwordx4 (set pointer register, byte offset = binding * 16). Set 2 bindings: 0 a,
1 b, 2 c, 3 d, 4 e, 5 f, 6 params, 7 extra. With exec-only barriers only params (6) and set-0 weights may be read
through the scalar cache.
usage: smemcheck.py ISA_FILE..."""
import re, sys
names = {0: 'a', 16: 'b', 32: 'c', 48: 'd', 64: 'e', 80: 'f', 96: 'params', 112: 'extra'}
for fn in sys.argv[1:]:
    text = open(fn).read()
    for prog in text.split('===== ')[1:]:
        name = prog.split()[0]
        # descriptor temps: %N:s[..] = s_load_dwordx4 %P:s[..], OFF ; set pointer temps: %P from s_mov of an input sgpr
        desc = {}
        movs = {}
        for m in re.finditer(r'%(\d+):s\[\d+(?:-\d+)?\] = s_mov_b32 %(\d+):s\[(\d+)\]', prog):
            movs[m.group(1)] = m.group(3)
        for m in re.finditer(r'%(\d+):s\[\d+-\d+\] = s_load_dwordx4 %(\d+):s\[(\d+)-\d+\], (0x[0-9a-f]+|\d+)', prog):
            off = int(m.group(4), 0)
            desc[m.group(1)] = (m.group(3), off)
        uses = {}
        for m in re.finditer(r's_buffer_load_\w+ %(\d+):s\[[\d-]+\]', prog):
            d = desc.get(m.group(1))
            key = f"base s{d[0]} +{d[1]:#x}" if d else f"%{m.group(1)}?"
            if d and d[1] in names:
                key += f" ({names[d[1]]})"
            uses[key] = uses.get(key, 0) + 1
        print(f"{name:36s} " + ', '.join(f"{k} x{v}" for k, v in sorted(uses.items())))
