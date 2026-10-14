#!/usr/bin/env python3
"""One field (a dotted path into the JSON line) of every gate-check step in DIR whose name starts with PREFIX.

usage: gate3-fields.py DIR PREFIX FIELD [FIELD...]
"""
import json
import pathlib
import sys

d, prefix, fields = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
for p in sorted(d.glob(f"{prefix}*.json")):
    text = p.read_text().strip()
    if not text:
        continue
    j = json.loads(text.splitlines()[-1])
    out = []
    for f in fields:
        v = j
        for k in f.split("."):
            v = v.get(k) if isinstance(v, dict) else None
        out.append(f"{f}={round(v, 3) if isinstance(v, float) else v}")
    print(p.stem, " ".join(out))
