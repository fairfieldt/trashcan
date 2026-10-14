#!/usr/bin/env python3
"""Shader statistics or disassembly by pipeline name from isa-dump2.sh logs (PIPELINE NAME lines before each shader).
usage: isa2.py stats LOG [NAME_SUBSTRING...]      (every pipeline when no names)
       isa2.py asm LOG NAME                       (disassembly of that pipeline)"""
import re, sys
mode, log = sys.argv[1], sys.argv[2]
parts = re.split(r'^PIPELINE (\S+)$', open(log).read(), flags=re.M)
for i in range(1, len(parts), 2):
    name, body = parts[i], parts[i + 1]
    if mode == 'stats':
        if len(sys.argv) > 3 and not any(s in name for s in sys.argv[3:]):
            continue
        g = lambda k: int(m.group(1)) if (m := re.search(k + r': (\d+)', body)) else -1
        print(f"{name:42s} code {g('Code size'):6d} lds {g('LDS size'):6d} vgpr {g('VGPRs'):3d} sgpr {g('SGPRs'):3d} "
              f"instr {g('Instructions'):5d} waves/simd {g('Subgroups per SIMD'):2d} latency {g('Latency'):6d} "
              f"vmem {g('VMEM'):4d} smem {g('SMEM'):4d} spills {g('Spilled VGPRs')}/{g('Spilled SGPRs')}")
    elif name == sys.argv[3]:
        d = body.split('\ndisasm:\n', 1)[1] if '\ndisasm:\n' in body else body
        print(d[:d.find('s_endpgm') + 9])
