#!/usr/bin/env python3
"""Count instruction opcodes in a line range of an ACO program dump. usage: opcount.py FILE FIRST LAST"""
import sys, re, collections
lines = open(sys.argv[1]).read().split('\n')[int(sys.argv[2]) - 1:int(sys.argv[3])]
c = collections.Counter()
for l in lines:
    m = re.search(r'\b((?:s|v|ds|buffer)_[a-z0-9_]+)', l)
    if m:
        c[m.group(1)] += 1
for k, v in c.most_common(16):
    print(f"{v:5d} {k}")
