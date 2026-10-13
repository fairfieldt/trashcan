#!/usr/bin/env python3
"""Name RADV shader stats (RADV_DEBUG=shaderstats) by pipeline creation order from engine/src/vulkan.rs.
usage: stats.py VULKAN_RS STATS_FILE [NAME_SUBSTRINGS...]"""
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
blocks = text.split('*** SHADER STATS ***')[1:]
print(f"{len(order)} pipelines, {len(blocks)} stats blocks", file=sys.stderr)
want = sys.argv[3:] if len(sys.argv) > 3 else None
for name, b in zip(order, blocks):
    if want and not any(w in name for w in want):
        continue
    g = lambda k: (re.search(rf'^{k}: (\d+)', b, re.M) or [None, '?'])[1]
    print(f"{name:34s} VGPR {g('VGPRs'):>3s} SGPR {g('SGPRs'):>3s} w/SIMD {g('Subgroups per SIMD'):>2s} "
          f"spill {g('Spilled VGPRs')}/{g('Spilled SGPRs')} LDS {g('LDS size'):>5s} instr {g('Instructions'):>5s} "
          f"VALU {g('VALU'):>4s} SALU {g('SALU'):>4s} VMEM {g('VMEM'):>3s} SMEM {g('SMEM'):>3s} lat {g('Latency'):>6s} ithr {g('Inverse Throughput'):>5s}")
