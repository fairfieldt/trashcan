#!/usr/bin/env python3
"""ISA audit of the decode kernels (results/qwen4/decode-kernels-20261014).

Which cached reads of data written earlier in the submission are allowed depends on the barrier that ends each
decode dispatch. Every buffer a dispatch of the graph writes is bound in descriptor set 2 (bindings a..f = 0..5,
extra = 7; the params buffer, binding 6, is written only by the graph's leading copy, which keeps a full barrier).
  --rule smem (default; the kept barrier, shader write -> none: RADV on GFX6 writes back and invalidates the L2 and
      invalidates the vector L1, but not the scalar cache): no scalar (s_buffer_load) loads from set 2 other than
      params.
  --rule coherent (execution-only barriers, measured and not kept: neither cache is invalidated): in addition, no
      vector loads from set 2 other than params without glc.
Set 0 holds weights (written only at load time) and may use either cache. Any access whose descriptor cannot be
traced fails.

For each pipeline the script traces every buffer access of the final ACO program (RADV_DEBUG=shaders on Tahiti, from
tools/vmstats.sh) to its descriptor (set, binding). RADV passes one 32-bit user SGPR per used descriptor set, in set
order; a descriptor is an s_load_dwordx4 at (set pointer, 16 * binding). Register copies are followed linearly
through the program.
usage: coherence_audit.py [--rule smem|coherent] VULKAN_RS RADV_DUMP [PIPELINE... | --scalar-safe]
(default: every pipeline; --scalar-safe: the engine's SCALAR_SAFE list in VULKAN_RS, the kernels that run after light
barriers; exit 1 on a violation)"""
import re, sys

args = sys.argv[1:]
rule = 'smem'
if args[:1] == ['--rule']:
    rule, args = args[1], args[2:]
assert rule in ('smem', 'coherent'), rule
src = open(args[0]).read()
start = src.index('for (name, bytes) in [')
end = src.index('] {', start)
names = re.findall(r'\(\s*"([A-Za-z0-9_]+)",\s*include_bytes!', src[start:end])
vstart = src.index("let variants: &[(&'static str, Option<u32>)] = match name {")
vend = src.index('_ => &[(name, None)],', vstart)
variants = {}
for m in re.finditer(r'"([A-Za-z0-9_]+)" => &\[(.*?)\],', src[vstart:vend], re.S):
    variants[m.group(1)] = re.findall(r'\("([A-Za-z0-9_]+)",', m.group(2))
order = []
for n in names:
    order += variants.get(n, [n])
parts = open(args[1]).read().split('*** SHADER STATS ***')
want = set(args[2:])
if want == {'--scalar-safe'}:
    a = src.index('const SCALAR_SAFE: &[&str] = &[')
    want = set(re.findall(r'"([A-Za-z0-9_]+)"', src[a:src.index('];', a)]))
BIND = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'params', 7: 'extra'}


def regs(spec):
    """'s[4-7]' -> [4, 5, 6, 7]; 's[3]' -> [3]."""
    m = re.match(r's\[(\d+)(?:-(\d+))?\]', spec)
    a = int(m.group(1))
    return list(range(a, int(m.group(2) or a) + 1))


def audit(prog):
    lines = prog.split('\n')
    origin = {}  # sgpr -> ('in', n) | ('const', v) | None
    tag = {}  # sgpr -> (base input sgpr, byte offset) of the descriptor it holds
    bases = set()
    accesses = []  # (kind, base, offset, glc, text)
    for line in lines:
        s = line.strip()
        if 'storage:vgpr_spill' in s or 'storage:scratch' in s:  # register spills, not buffer accesses
            continue
        m = re.match(r'(?:\w+: )?%\d+:(s\[[\d-]+\])(?:,\s+\w+: %\d+:\w+)? = (\w+)(?: (.*))?$', s)
        if not m:
            m2 = re.match(r'(buffer_\w+|s_buffer_\w+) (.*)', s)
            if m2:  # stores (no destination)
                op, rest = m2.groups()
                d = re.search(r'%\d+:(s\[[\d-]+\])', rest)
                if d and op.startswith('buffer_store') or (d and op.startswith('buffer_atomic')):
                    r0 = regs(d.group(1))[0]
                    accesses.append((op, *(tag.get(r0) or (None, None)), ' glc' in rest, s))
            else:
                m3 = re.match(r'(?:\w+: )?%\d+:(v\[[\d-]+\]) = (buffer_\w+) (.*)', s)
                if m3:
                    op, rest = m3.group(2), m3.group(3)
                    d = re.search(r'%\d+:(s\[[\d-]+\])', rest)
                    if d:
                        r0 = regs(d.group(1))[0]
                        accesses.append((op, *(tag.get(r0) or (None, None)), ' glc' in rest, s))
            continue
        dst, op, rest = m.groups()
        rest = rest or ''
        dr = regs(dst)
        srcs = re.findall(r'%\d+:(s\[[\d-]+\])', rest)
        if op == 's_load_dwordx4' and srcs:
            br = regs(srcs[0])
            off = re.search(r', (0x[0-9a-f]+|\d+)\s*(?:storage|$)', rest + ' ')
            off = int(off.group(1), 0) if off else 0
            o = origin.get(br[0], ('in', br[0]))
            if o and o[0] == 'pc':
                for r in dr:
                    tag[r] = ('pc', 0)
                continue
            if o and o[0] == 'in':
                bases.add(o[1])
                for i, r in enumerate(dr):
                    tag[r] = (o[1], off)
            else:
                for r in dr:
                    tag.pop(r, None)
            for r in dr:
                origin[r] = None
            continue
        if op.startswith('s_buffer_load') and srcs:
            r0 = regs(srcs[0])[0]
            accesses.append((op, *(tag.get(r0) or (None, None)), False, s))
        if op == 's_getpc_b64':
            for r in dr:
                origin[r] = ('pc', 0)
                tag[r] = ('pc', 0)
            continue
        if op in ('s_mov_b32', 's_mov_b64') and srcs:
            sr = regs(srcs[0])
            for d_, s_ in zip(dr, sr):
                origin[d_] = origin.get(s_, ('in', s_))
                if s_ in tag:
                    tag[d_] = tag[s_]
                else:
                    tag.pop(d_, None)
            continue
        if op in ('s_mov_b32',) and not srcs:
            for r in dr:
                origin[r] = ('const', rest.split()[0])
                tag.pop(r, None)
            continue
        # Arithmetic on a PC-relative address (the constant data's offset) keeps it PC-relative.
        pc = any(tag.get(r) == ('pc', 0) for sp in srcs for r in regs(sp))
        for r in dr:
            origin[r] = ('pc', 0) if pc else None
            if pc:
                tag[r] = ('pc', 0)
            else:
                tag.pop(r, None)
    # Set pointers are the bases of the descriptors the program actually uses as buffer resources (push constants are
    # loaded with the same instructions from another pointer). Set 0 holds one binding, so a base with descriptors past
    # offset 0 is set 2; with two bases, the other is set 0.
    used = {}
    for op, base, off, glc, text in accesses:
        if base is not None and base != 'pc':
            used.setdefault(base, set()).add(off)
    if len(used) == 1:
        setof = {b: 2 for b in used}
    else:
        setof = {b: (2 if any(o > 0 for o in offs) else 0) for b, offs in used.items()}
    report, bad = {}, []
    for op, base, off, glc, text in accesses:
        if base is None:
            key = 'untraced'
        elif base == 'pc':
            key = 'const'
        else:
            st = setof.get(base, '?')
            key = f"set{st}:{BIND.get(off // 16, off) if st == 2 else off // 16}"
        kind = 'smem' if op.startswith('s_') else ('store' if 'store' in op or 'atomic' in op else 'load')
        report.setdefault(key, {}).setdefault((kind, glc), 0)
        report[key][(kind, glc)] += 1
        if key.startswith('set2:') and key != 'set2:params':
            if kind == 'smem' or (rule == 'coherent' and kind == 'load' and not glc):
                bad.append(text)
        if key.startswith('set?'):
            bad.append('unknown set: ' + text)
        if key == 'untraced' and kind != 'store':
            bad.append('untraced: ' + text)
    return report, bad


fail = 0
missing = want - set(order)
if missing:
    print('not in the dump:', ' '.join(sorted(missing)))
    fail = 1
for i, name in enumerate(order):
    if want and name not in want:
        continue
    k = parts[i].rfind('falling back to print_program.')
    prog = parts[i][k:] if k >= 0 else parts[i]
    report, bad = audit(prog)
    cols = []
    for key in sorted(report):
        r = report[key]
        cols.append(key + ' ' + ' '.join(f"{kind}{'/glc' if glc else ''} {n}" for (kind, glc), n in sorted(r.items())))
    print(f"{'FAIL' if bad else 'ok  '} {name:36s} " + '; '.join(cols))
    for b in bad[:6]:
        print(f"       {b[:150]}")
    fail |= bool(bad)
sys.exit(fail)
