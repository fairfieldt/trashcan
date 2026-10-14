#!/usr/bin/env python3
"""Summarize run.sh's xgpu outputs per kernel directory: p2p-verify results, and from p2p-wall the P2P write
(CPU submit-to-fence) at 1, 8 and 32 MiB and the P2P read at 32 MiB.
usage: p2psum.py DIR [DIR ...]"""
import glob
import os
import re
import sys

for d in sys.argv[1:]:
    verify = []
    for f in sorted(glob.glob(os.path.join(d, 'xgpu-verify-*.txt'))):
        text = open(f).read()
        bad = re.findall(r'(\d+) MiB checked, (\d+) bad words', text)
        verify.append('PASS' if 'p2p-verify: PASS' in text else 'not run' if 'exit 143' in text else 'FAIL')
        verify[-1] += ' (' + ', '.join(f'{m} MiB {b} bad' for m, b in bad) + ')' if bad else ''
    wall = open(os.path.join(d, 'xgpu-p2p-wall.txt')).read()
    fence = dict(re.findall(r'CPU submit-to-fence\s+(\d+) KiB\s+[\d.]+ us\s+([\d.]+) GB/s', wall))
    read = dict(re.findall(r'\(P2P read\)\s+(\d+) KiB\s+[\d.]+ us\s+([\d.]+) GB/s', wall))
    print(os.path.basename(os.path.normpath(d)))
    for i, v in enumerate(verify, 1):
        print(f'  p2p-verify {i}: {v}')
    print('  P2P write, submit-to-fence, 1 / 8 / 32 MiB: ' + ' / '.join(fence.get(k, '?') for k in ('1024', '8192', '32768')) + ' GB/s')
    print(f"  P2P read, 32 MiB: {read.get('32768', '?')} GB/s")
