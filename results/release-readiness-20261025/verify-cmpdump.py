#!/usr/bin/env python3
"""Negative fixtures for the mixer comparison gate; no GPU or model required."""
from array import array
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

script = Path(__file__).resolve().parents[1] / 'qwen4/mixer-copies-20261023/tools/cmpdump.py'
with TemporaryDirectory() as directory:
    candidate, reference = Path(directory) / 'candidate', Path(directory) / 'reference'
    candidate.mkdir()
    reference.mkdir()
    index = reference / 'index.tsv'
    index.write_text('tensor\tfixture\n')
    (reference / 'tensor.f32').write_bytes(array('f', [1, 2]).tobytes())
    for label, values, expected in [
        ('identical', [1, 2], 0), ('changed', [1, 3], 1),
        ('truncated', [1], 1), ('missing', None, 1), ('empty reference', None, 1),
    ]:
        tensor = candidate / 'tensor.f32'
        if values is None:
            tensor.unlink(missing_ok=True)
        else:
            tensor.write_bytes(array('f', values).tobytes())
        if label == 'empty reference':
            index.write_text('')
        result = subprocess.run([sys.executable, str(script), str(candidate), str(reference)],
                                capture_output=True, text=True)
        if result.returncode != expected:
            raise SystemExit(f'{label}: expected {expected}, got {result.returncode}\n{result.stdout}{result.stderr}')
        print(f'{label}: exit {result.returncode}, expected {expected}')
