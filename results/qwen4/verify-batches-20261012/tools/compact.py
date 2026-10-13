#!/usr/bin/env python3
"""Compact cmp.py output: each cell as its short name and microseconds."""
import sys
for line in sys.stdin:
    cells = []
    for cell in line.rstrip('\n').split('|'):
        parts = cell.split()
        if not parts:
            cells.append(' ' * 26)
            continue
        cells.append(f"{parts[0][:18]:<18} {parts[-1]:>7}")
    print(' | '.join(cells))
