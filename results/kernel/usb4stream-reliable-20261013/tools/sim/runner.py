#!/usr/bin/env python3
"""Stand-in for q4run.py and tprun.py in a window script's simulation (simtest.sh): sleeps briefly and prints a
driver result line."""
import json, os, sys, time

with open(os.environ.get('SIMLOG', '/dev/null'), 'a') as f:
    f.write(f'{time.time():.3f} runner: {" ".join(sys.argv[1:])}\n')
time.sleep(float(os.environ.get('SIMRUN', '0.5')))
print(json.dumps({'pp': 512, 'tg': 128, 'pp_tok_s': 500.0, 'tg_tok_s': 21.0}))
