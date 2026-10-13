#!/usr/bin/env python3
"""Compare dumps against a reference with compare_activations.py, save the full comparison next to each dump as
DUMP-vs-TAG.txt, and print the logits line, the result norm, a few residuals and the greedy summary.
usage: vsref.py REF TAG DUMP [DUMP ...]"""
import subprocess, sys, os, re
TOOL = '/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a2c014fa1f29b2375/tools/compare_activations.py'
ref, tag = sys.argv[1], sys.argv[2]
keys = re.compile(r'^(logits|result_norm|result_output|l_last-47|hc_combine-(11|19|23|27|47)) ')
for d in sys.argv[3:]:
    out = subprocess.run(['python3', TOOL, d, ref, '--all'], capture_output=True, text=True).stdout
    open(f"{d.rstrip('/')}-vs-{tag}.txt", 'w').write(out)
    print(f"== {os.path.basename(d.rstrip('/'))} vs {tag}")
    for l in out.split('\n'):
        if keys.match(l) or l.startswith('worst') or l.startswith('greedy') or l.startswith('logits:'):
            print('  ' + l)
