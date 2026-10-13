#!/usr/bin/env python3
"""Pipelines of a RADV_DEBUG=shaders[,shaderstats] log named by creation order (engine/src/vulkan.rs of the worktree).
usage: isa.py LOG [NAME ...]   stats of every pipeline (or those containing a NAME); --dump NAME prints its program"""
import re, sys
src = open('/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a2c014fa1f29b2375/engine/src/vulkan.rs').read()
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
text = open(sys.argv[1]).read()
parts = text.split('\nCompute Shader\ndisasm:\n')[1:]
stats = text.split('*** SHADER STATS ***')[1:]
print(f"{len(order)} pipelines, {len(parts)} programs, {len(stats)} stats", file=sys.stderr)
args = sys.argv[2:]
if args and args[0] == '--dump':
    for name, body in zip(order, parts):
        if name == args[1]:
            cut = body.find('shader: MESA_SHADER')
            print(body[:cut if cut >= 0 else len(body)])
    sys.exit()
for i, (name, body) in enumerate(zip(order, parts)):
    if args and not any(a in name for a in args):
        continue
    st = stats[i] if i < len(stats) else ''
    g = lambda k: (re.search(rf'^{k}: (\d+)', st, re.M) or [None, '?'])[1]
    cut = body.find('shader: MESA_SHADER')
    prog = body[:cut if cut >= 0 else len(body)]
    ops = re.findall(r'^\s+(?:[^=\n]*= )?([a-z][a-z_0-9]+)', prog, re.M)
    cnt = lambda pre: sum(1 for o in ops if o.startswith(pre))
    print(f"{name:34s} VGPRs {g('VGPRs'):>3s} SGPRs {g('SGPRs'):>3s} waves/SIMD {g('Subgroups per SIMD'):>2s} "
          f"scratch {g('Scratch size'):>4s} spillV {g('Spilled VGPRs')} LDS {g('LDS size'):>5s} instrs {g('Instructions'):>5s} "
          f"| v_ {cnt('v_')} s_buffer_load {cnt('s_buffer_load')} s_load {cnt('s_load')} buffer_load {cnt('buffer_load')} "
          f"ds_ {cnt('ds_')} s_waitcnt {cnt('s_waitcnt')}")
