#!/usr/bin/env python3
"""Stub of deadpath.py on a simulated Mac: waits for its schedule, then reports a working cable."""
import json, os, sys, time

role, dev, start, k, t_open, t_test, t_end, run = sys.argv[1:9]
while time.time() < float(start) + float(t_end):
    time.sleep(0.05)
out = {'role': role, 'dev': dev, 'k': int(k), 'open': float(t_open), 'run': int(run), 'host': os.environ.get('SIMHOST'),
       'test_written': 20, 'burst_read': 0, 'test_read': 20, 'other': 0, 'eofs': 0}
if role == 'early':
    out['burst_written'] = int(k)
print(json.dumps(out))
