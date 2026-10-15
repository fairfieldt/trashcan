#!/usr/bin/env python3
"""Compare two validation runs bit for bit: every file of the dump directories (captured tensors, greedy.txt) and the
greedy ids the driver printed (`greedy: [...]` in RUN.json). Prints one line; exits 1 when anything differs.

usage: dg-valcmp.py REF_DIR CAND_DIR   (the runs' .json files next to the directories)
"""
import pathlib
import sys

a, b = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
fa = sorted(p.name for p in a.iterdir()) if a.is_dir() else []
fb = sorted(p.name for p in b.iterdir()) if b.is_dir() else []
diff = [f for f in fa if f not in fb or (a / f).read_bytes() != (b / f).read_bytes()] + [f for f in fb if f not in fa]


def greedy(d):
    j = d.with_name(d.name + ".json")
    return [l for l in j.read_text().splitlines() if l.startswith("greedy:")] if j.exists() else []


ga, gb = greedy(a), greedy(b)
ok = fa and not diff and ga and ga == gb
print(f"{b.name} against {a.name}: {len(fa)} files, greedy {'same' if ga == gb else 'DIFFERENT'}: "
      f"{'identical' if ok else 'DIFFER ' + str(diff[:5]) + (' (no greedy line)' if not ga else '')}")
sys.exit(0 if ok else 1)
