#!/usr/bin/env python3
"""Replay and check a disposable regular image without a prior lwext4 remount."""
from pathlib import Path
import os, shutil, subprocess, sys
image = Path(sys.argv[1])
assert image.is_file(), "a disposable regular image file is required"
fsck = os.environ.get("E2FSCK", shutil.which("e2fsck") or "/opt/homebrew/opt/e2fsprogs/sbin/e2fsck")
env = dict(os.environ, LC_ALL="C")
for args in (["-E", "journal_only", "-y"], ["-fn"]):
    result = subprocess.run([fsck, *args, str(image)], text=True, capture_output=True, env=env)
    output = result.stdout + result.stderr
    print(output, end="")
    assert result.returncode == 0, (args, result.returncode)
    # e2fsck can exit zero despite an unfixed summary-count mismatch. Do not
    # let a repairing check or a later writable mount hide that diagnostic.
    for bad in ["Fix?", "Clear?", "Salvage?", "count wrong", "bitmap differences", "has zero dtime", "UNEXPECTED INCONSISTENCY"]:
        assert bad.lower() not in output.lower(), (args, bad)
