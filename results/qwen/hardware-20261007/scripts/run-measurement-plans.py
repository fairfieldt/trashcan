#!/usr/bin/env python3
"""Run independent hardware plans in order; stop on incomplete work or failed checks."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
for name in sys.argv[1:]:
    path = root / (name + '-plan.json')
    if not path.is_file():
        raise SystemExit('missing plan: ' + str(path))
    command = [sys.executable, str(root / 'scripts/cooled-plan.py'), str(path),
               '--max-c', '65', '--max-plan-minutes', '60']
    print('Starting ' + name, flush=True)
    result = subprocess.run(command)
    summary = json.loads((root / 'plans' / path.stem / 'summary.json').read_text())
    if result.returncode or summary.get('state') != 'completed' or summary.get('active_tickets'):
        raise SystemExit('Stopped after ' + name + '; inspect summary and active tickets')
    subprocess.run([sys.executable, str(root / 'refresh-analysis.py')], check=True,
                   stdout=subprocess.DEVNULL)
    analysis = json.loads((root / 'analysis-summary.json').read_text())
    failures = [(pair.get('baseline'), pair.get('candidate'))
                for pair in analysis.get('comparisons', [])
                if any(not row.get('correctness_pass', True) for row in pair.get('pairs', []))]
    if failures:
        raise SystemExit('Correctness comparison failed: ' + repr(failures))
