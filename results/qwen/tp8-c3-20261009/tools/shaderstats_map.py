#!/usr/bin/env python3
"""Name the pipelines in a RADV_DEBUG=shaderstats dump of the engine.

Pipelines are created in the order of the shader list in engine/src/vulkan.rs (attention and other variants
expanded as in its `variants` match), so the N-th stats block belongs to the N-th name.
usage: shaderstats_map.py VULKAN_RS < stats.txt [NAME ...]   (NAME: only print these; default all)
"""
import re, sys

src = open(sys.argv[1]).read()
names = re.findall(r'\(\s*"([A-Za-z0-9_]+)",\s*include_bytes!\(concat!\(env!\("OUT_DIR"\)', src)
# variants: `"name" => &[ ("v1", ..), ("v2", ..) ]`
variants = {}
for m in re.finditer(r'"([A-Za-z0-9_]+)" => &\[(.*?)\],\n', src, re.S):
    variants[m.group(1)] = re.findall(r'\("([A-Za-z0-9_]+)",', m.group(2))
order = []
for n in names:
    order += variants.get(n, [n])
want = set(sys.argv[2:])
text = sys.stdin.read()
blocks = re.split(r'\n(?=\S.*Shader:|\n*Compute Shader)', text)
stats = []
cur = None
for line in text.splitlines():
    if line.startswith("Compute Shader"):
        cur = {}
        stats.append(cur)
    elif cur is not None:
        m = re.match(r'^\s*([A-Za-z ]+?):\s*(\S+)', line)
        if m:
            cur[m.group(1).strip()] = m.group(2)
if len(stats) != len(order):
    print(f'warning: {len(stats)} stats blocks, {len(order)} pipeline names', file=sys.stderr)
for name, st in zip(order, stats):
    if want and name not in want:
        continue
    print(name, ' '.join(f'{k}={v}' for k, v in st.items()))
