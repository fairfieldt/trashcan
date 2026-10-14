#!/usr/bin/env python3
"""Stub of holdopen.py on a simulated Mac: marks the devices open (a file the pathdump stub reads) for SECONDS."""
import json, os, sys, time

secs, devs = float(sys.argv[1]), sys.argv[2:]
mark = os.path.join(os.environ['HOME'], 'holding')
open(mark, 'w').write(' '.join(devs))
try:
    time.sleep(secs)
finally:
    os.remove(mark)
print(json.dumps({'step': 'holdopen', 'host': os.environ.get('SIMHOST'), 'secs': secs,
                  'devs': [{'dev': d, 'opened': True, 'open_ms': 0.4, 'bytes': 0, 'eofs': 0, 'errors': {},
                            'close_ms': 0.5} for d in devs]}))
