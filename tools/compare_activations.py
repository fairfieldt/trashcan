#!/usr/bin/env python3
"""Compare two activation dumps (the format written by tools/dump-activations): per tensor, the relative L2 error
||a - b|| / ||b|| and cosine similarity of the last token's row, plus top-k agreement of the logits.

usage: compare_activations.py DIR_A DIR_B [--all]

DIR_B is the reference. Without --all, only l_out-*, attn_out-*, result_norm and the logits are shown.
No numpy needed.
"""
import argparse
import math
import os
import sys
from array import array


def load(path):
    a = array("f")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    return a


def metrics(a, b):
    if len(a) != len(b) or not all(math.isfinite(x) for x in (*a, *b)):
        return float("inf"), float("nan")
    num = den = dot = na = 0.0
    for x, y in zip(a, b):
        d = x - y
        num += d * d
        den += y * y
        dot += x * y
        na += x * x
    rel = math.sqrt(num / den) if den > 0 else (0.0 if num == 0 else float("inf"))
    cos = dot / math.sqrt(na * den) if na > 0 and den > 0 else (1.0 if na == den == 0 else float("nan"))
    return rel, cos


def topk(v, k):
    return sorted(range(len(v)), key=lambda i: -v[i])[:k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual")
    parser.add_argument("reference")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--check", action="store_true", help="fail on missing required tensors or tolerance/greedy failures")
    parser.add_argument("--max-rel", type=float, default=0.04)
    parser.add_argument("--min-cos", type=float, default=0.999)
    parser.add_argument("--greedy", type=int, default=10)
    opts = parser.parse_args()
    da, db, show_all = opts.actual, opts.reference, opts.all
    failures = []
    names = [line.split("\t")[0] for line in open(os.path.join(db, "index.tsv"))]
    names = list(dict.fromkeys(n for n in names if "(reshaped)" not in n and "(view)" not in n and not n.startswith(("norm-", "kq-", "kq_soft_max-", "kqv-"))))
    worst = 0.0
    for name in names:
        brief = name.startswith("l_out-") or name.startswith("attn_out-") or name == "result_norm"
        pa, pb = os.path.join(da, name + ".f32"), os.path.join(db, name + ".f32")
        if not os.path.exists(pa):
            failures.append(f"missing {name}")
            if show_all or brief:
                print(f"{name:32} missing in {da}")
            continue
        a, b = load(pa), load(pb)
        if len(a) != len(b):
            failures.append(f"length mismatch {name}")
            print(f"{name:32} length {len(a)} vs {len(b)}")
            continue
        rel, cos = metrics(a, b)
        worst = max(worst, rel)
        if not math.isfinite(rel) or rel > opts.max_rel or not math.isfinite(cos) or cos < opts.min_cos:
            failures.append(f"tolerance {name}: rel {rel:.3e}, cos {cos:.6f}")
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
    if len(la) != len(lb) or rel > opts.max_rel or cos < opts.min_cos or not math.isfinite(rel) or not math.isfinite(cos):
        failures.append("logit tolerance or length mismatch")
    if ta[0] != tb[0]:
        failures.append("top-1 mismatch")
    if same < opts.greedy:
        failures.append(f"greedy match {same} < {opts.greedy}")
    if opts.check:
        if failures:
            print("FAILED: " + "; ".join(failures), file=sys.stderr)
            return 1
        print(f"PASS: {len(names)} required tensors, logits, and {opts.greedy} greedy tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
