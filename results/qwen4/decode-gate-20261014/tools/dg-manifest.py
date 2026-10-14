#!/usr/bin/env python3
"""SHA-256 of every validation dump (*.f32) of each run directory, written to RUN/dumps.sha256 (sorted by name), so that
the dumps themselves can stay out of git: runs with identical manifests have bit-identical dumps.

usage: dg-manifest.py RUN_DIR...
"""
import hashlib
import pathlib
import sys

for arg in sys.argv[1:]:
    d = pathlib.Path(arg)
    dumps = sorted(d.glob("*.f32"))
    if not dumps:
        continue
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in dumps]
    (d / "dumps.sha256").write_text("\n".join(lines) + "\n")
    print(f"{d}: {len(dumps)} dumps")
