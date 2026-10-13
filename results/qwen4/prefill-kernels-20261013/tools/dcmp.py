#!/usr/bin/env python3
"""Summarize compare_activations.py between dump directories: per pair, the number of tensors, how many are
bit-identical, the largest relative error and its tensor, and the errors of a few key tensors.
usage: dcmp.py A:B [A:B ...]   (B is the reference)"""
import subprocess, sys, re
TOOL = '/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a2c014fa1f29b2375/tools/compare_activations.py'
KEYS = ('attn_output-0', 'linear_attn_out-0', 'ffn_out-0', 'hc_mixed-0', 'l_last-7', 'hc_combine-7', 'attn_output-7')
for pair in sys.argv[1:]:
    a, b = pair.split(':')
    out = subprocess.run(['python3', TOOL, a, b, '--all'], capture_output=True, text=True).stdout
    rows = re.findall(r'^(\S+)\s+rel\s+(\S+)\s+cos\s+(\S+)', out, re.M)
    rel = {n: float(r) for n, r, _ in rows}
    worst = max(rel.items(), key=lambda x: x[1]) if rel else ('-', 0)
    zero = sum(1 for v in rel.values() if v == 0)
    print(f"{a.split('/')[-1]} vs {b.split('/')[-1]}: {len(rel)} tensors, {zero} identical, max rel {worst[1]:.2e} "
          f"({worst[0]}); " + ', '.join(f"{k} {rel[k]:.1e}" for k in KEYS if k in rel))
