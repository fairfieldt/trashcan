#!/usr/bin/env python3
"""One A/B pair of the six-Mac session: the rates of the base and candidate runs (`bench`: tg_tok_s; `mtp spec`: the
mean MTP tok/s over the prompts, and the plain tok/s beside it) and the candidate's difference. Exits 1 when the
difference is under 3% (another pair is due), else 0.

usage: dg-pair.py BASE.json CAND.json
"""
import json
import statistics
import sys


def rates(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    if rows and "tg_tok_s" in rows[0]:
        return rows[0]["tg_tok_s"], None
    mtp = statistics.mean(r["mtp"]["tok_s"] for r in rows)
    plain = statistics.mean(r["plain_tok_s"] for r in rows)
    return mtp, plain


(b, bp), (c, cp) = rates(sys.argv[1]), rates(sys.argv[2])
d = c / b - 1
extra = f"; plain {bp:.2f} -> {cp:.2f} tok/s ({100 * (cp / bp - 1):+.2f}%)" if bp else ""
print(f"{b:.2f} -> {c:.2f} tok/s ({100 * d:+.2f}%){extra}")
sys.exit(0 if abs(d) >= 0.03 else 1)
