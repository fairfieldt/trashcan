#!/usr/bin/env python3
"""Print the final ACO program (RADV_DEBUG=shaders on the null device) of the named pipelines.
usage: isa.py VULKAN_RS STATS_FILE NAME [NAME...]   (exact pipeline names)"""
import re, sys
src = open(sys.argv[1]).read()
start = src.index('for (name, bytes) in [')
end = src.index('] {', start)
names = re.findall(r'\(\s*"([A-Za-z0-9_]+)",\s*include_bytes!', src[start:end])
vstart = src.index('let variants: &[(&\'static str, Option<u32>)] = match name {')
vend = src.index('_ => &[(name, None)],', vstart)
variants = {}
for m in re.finditer(r'"([A-Za-z0-9_]+)" => &\[(.*?)\],', src[vstart:vend], re.S):
    variants[m.group(1)] = re.findall(r'\("([A-Za-z0-9_]+)",', m.group(2))
order = []
for n in names:
    order += variants.get(n, [n])
text = open(sys.argv[2]).read()
# Each pipeline: "...disasm:\n<program>\nCompute Shader:\n*** SHADER STATS ***"
parts = text.split('*** SHADER STATS ***')
for i, name in enumerate(order):
    if name not in sys.argv[3:]:
        continue
    before = parts[i]
    k = before.rfind('falling back to print_program.')
    prog = before[k:] if k >= 0 else before[-20000:]
    print(f"===== {name}")
    print(prog)
