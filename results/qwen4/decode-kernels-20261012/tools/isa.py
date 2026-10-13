#!/usr/bin/env python3
"""Extract the disassembly of named pipelines from a RADV_DEBUG=shaders,nocache log (creation order of vulkan.rs).
usage: isa.py LOG NAME [NAME...]"""
import re, sys
src = open('/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a90072f31ff42427f/engine/src/vulkan.rs').read()
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
text = open(sys.argv[1]).read()
parts = text.split('\nCompute Shader\ndisasm:\n')[1:]
print(f"{len(order)} pipelines, {len(parts)} shaders", file=sys.stderr)
for name, body in zip(order, parts):
    if name in sys.argv[2:]:
        end = body.find('s_endpgm')
        print(f"===== {name}")
        print(body[:end + 9])
