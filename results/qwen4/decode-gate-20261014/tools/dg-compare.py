#!/usr/bin/env python3
"""Compare the branch's (B) runs of dg-correct.sh with main's (A), bit for bit: validation dumps (every .f32 file and
greedy.txt) and the token ids of `mtp spec` (plain and MTP decoding). Prints one line per pair and exits nonzero when
anything differs.

usage: dg-compare.py RESULTS_DIR
"""
import json
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
bad = 0
for a in sorted(d.glob("*-A")):
    if not a.is_dir():
        continue
    b = d / (a.name[:-2] + "-B")
    files = sorted(p.name for p in a.iterdir())
    other = sorted(p.name for p in b.iterdir()) if b.is_dir() else []
    diff = [f for f in files if f not in other or (a / f).read_bytes() != (b / f).read_bytes()]
    missing = [f for f in other if f not in files]
    tokens = len((a / "greedy.txt").read_text().split()) if (a / "greedy.txt").exists() else 0
    ok = not diff and not missing and files
    bad += not ok
    print(f"{a.name[:-2]:12} {len(files)} files, {tokens} greedy tokens: {'identical' if ok else f'DIFFER {diff[:5]} {missing[:5]}'}")


def spec_runs(path):
    runs = []
    for line in path.read_text().splitlines():
        try:
            j = json.loads(line)
        except ValueError:
            continue
        if "mtp" in j:
            runs.append(j)
    return runs


for a in sorted(d.glob("mtp-*-A.out")):
    b = d / (a.name[:-6] + "-B.out")
    ra, rb = spec_runs(a), spec_runs(b) if b.exists() else []
    same = len(ra) == len(rb) > 0 and all(x["mtp"]["ids"] == y["mtp"]["ids"] for x, y in zip(ra, rb))
    agree = all(r["mtp"].get("identical") for r in ra + rb)
    bad += not (same and agree)
    stats = "; ".join(
        f"{k}: accepted per draft {r['mtp'].get('accepted_per_draft')}, {len(r['mtp']['ids'])} ids"
        for k, r in (("A", ra[0] if ra else {"mtp": {"ids": []}}), ("B", rb[0] if rb else {"mtp": {"ids": []}}))
    )
    print(f"{a.name[:-6]:12} MTP ids {'identical' if same else 'DIFFER'}, plain and MTP agree: {agree} ({stats})")
sys.exit(1 if bad else 0)
