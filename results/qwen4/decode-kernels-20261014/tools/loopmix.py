#!/usr/bin/env python3
"""Instruction mix of each loop body (loop-header .. loop-exit) in isa.py output.
usage: loopmix.py ISA_FILE"""
import re, sys, collections
name, inloop, mix, n = None, False, collections.Counter(), 0
def flush():
    global mix
    if mix:
        tot = sum(mix.values())
        valu = sum(v for k, v in mix.items() if k.startswith('v_') and not k.startswith('v_readfirstlane'))
        print(f"{name} loop {n}: {tot} instrs, VALU {valu}: " + ", ".join(f"{k} {v}" for k, v in mix.most_common(14)))
    mix = collections.Counter()
for line in open(sys.argv[1]):
    if line.startswith('====='):
        flush(); name = line.split()[1]; n = 0; inloop = False; continue
    if 'kind:' in line and 'loop-header' in line:
        flush(); inloop = True; n += 1; continue
    if 'kind:' in line and 'loop-exit' in line:
        flush(); inloop = False; continue
    if not inloop:
        continue
    m = re.match(r'\s+(?:[vs]\d+: [^=]*= )?([a-z_0-9]+)', line)
    if m and (m.group(1).startswith(('v_', 's_', 'ds_', 'buffer_'))):
        mix[m.group(1)] += 1
flush()
