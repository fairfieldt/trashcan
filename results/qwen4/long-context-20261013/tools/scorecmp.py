#!/usr/bin/env python3
"""Compare the block scores of two dumps bit for bit (indexer_scores-L, every token's row; TRASHCAN_Q4_DUMP_ALL=1):
only the complete blocks b < nb(t) of tokens past the dense regime are compared.
usage: scorecmp.py DIR_A DIR_B PROMPT_TOKENS [--layer 3] [--stride 8192] [--kind 4] [--m 512]"""
import sys
from array import array

a, b, ntok = sys.argv[1], sys.argv[2], int(sys.argv[3])
opt = lambda k, d: int(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
layer, stride, kind, m = opt('--layer', 3), opt('--stride', 8192), opt('--kind', 4), opt('--m', 512)
def load(d):
    x = array('I')
    with open(f"{d}/indexer_scores-{layer}.f32", 'rb') as f:
        x.frombytes(f.read())
    return x
x, y = load(a), load(b)
rows = len(y) // stride
first = ntok - rows
checked = same = 0
worst = None
for t in range(rows):
    nb = (first + t + 1) // kind
    if nb <= m:
        continue
    ra, rb = x[t * stride:t * stride + nb], y[t * stride:t * stride + nb]
    d = sum(1 for p, q in zip(ra, rb) if p != q)
    checked += nb
    same += nb - d
    if d and worst is None:
        worst = (t, d)
print(f"layer {layer}: {rows} tokens from position {first}, {checked} scores compared, {checked - same} differ"
      + (f" (first token with differences: {worst[0]}, {worst[1]} scores)" if worst else ""))
