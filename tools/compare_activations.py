#!/usr/bin/env python3
"""Compare two activation dumps (the format written by tools/dump-activations): per tensor, the relative L2 error
||a - b|| / ||b|| and cosine similarity of the last token's row, plus top-k agreement of the logits.

usage: compare_activations.py DIR_A DIR_B [--all]

DIR_B is the reference. Without --all, only l_out-*, attn_out-*, result_norm and the logits are shown.
No numpy needed.
"""
import math
import os
import sys
from array import array


def load(path):
    a = array("f")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    return a


def metrics(a, b):
    num = den = dot = na = 0.0
    for x, y in zip(a, b):
        d = x - y
        num += d * d
        den += y * y
        dot += x * y
        na += x * x
    rel = math.sqrt(num / den) if den > 0 else float("inf")
    cos = dot / math.sqrt(na * den) if na > 0 and den > 0 else float("nan")
    return rel, cos


def topk(v, k):
    return sorted(range(len(v)), key=lambda i: -v[i])[:k]


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    show_all = "--all" in sys.argv
    da, db = args
    names = [line.split("\t")[0] for line in open(os.path.join(db, "index.tsv"))]
    worst = 0.0
    for name in names:
        brief = name.startswith("l_out-") or name.startswith("attn_out-") or name == "result_norm"
        pa, pb = os.path.join(da, name + ".f32"), os.path.join(db, name + ".f32")
        if not os.path.exists(pa):
            if show_all or brief:
                print(f"{name:32} missing in {da}")
            continue
        a, b = load(pa), load(pb)
        if len(a) != len(b):
            print(f"{name:32} length {len(a)} vs {len(b)}")
            continue
        rel, cos = metrics(a, b)
        worst = max(worst, rel)
        if show_all or brief:
            print(f"{name:32} rel {rel:9.2e}  cos {cos:.6f}")
    print(f"worst relative error over {len(names)} tensors: {worst:.2e}")

    la, lb = load(os.path.join(da, "logits.f32")), load(os.path.join(db, "logits.f32"))
    rel, cos = metrics(la, lb)
    ta, tb = topk(la, 10), topk(lb, 10)
    print(f"logits: rel {rel:.2e} cos {cos:.6f}; top-1 {ta[0]} vs {tb[0]}; top-10 overlap {len(set(ta) & set(tb))}/10")
    ga = [l.strip() for l in open(os.path.join(da, "greedy.txt"))] if os.path.exists(os.path.join(da, "greedy.txt")) else []
    gb = [l.strip() for l in open(os.path.join(db, "greedy.txt"))]
    same = 0
    for x, y in zip(ga, gb):
        if x != y:
            break
        same += 1
    print(f"greedy: first {same} of {len(gb)} tokens identical")


if __name__ == "__main__":
    main()
