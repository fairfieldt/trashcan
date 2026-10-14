#!/usr/bin/env python3
"""Stub of closecheck.py on a simulated Mac."""
import json, os, sys

host = os.environ.get('SIMHOST', '?')
if sys.argv[1] == 'plant':
    print(json.dumps({'step': 'plant', 'dev': sys.argv[2], 'close_ms': [500.0] * int(sys.argv[3]), 'host': host}))
else:
    print(json.dumps({'step': 'check', 'dev': sys.argv[2], 'bytes': 0, 'eofs': 0, 'write': 'ok', 'bytes_after': 0,
                      'eofs_after': 0, 'sticky': False, 'close_ms': 500.0, 'host': host}))
