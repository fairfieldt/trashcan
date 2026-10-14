#!/usr/bin/env python3
"""Stub of udpping.py on a simulated Mac."""
import json, os, sys, time

host = os.environ.get('SIMHOST', '?')
if sys.argv[1] == 'server':
    time.sleep(1.5)
    print(json.dumps({'role': 'server', 'addr': sys.argv[2], 'echoed': 2000, 'host': host}))
else:
    time.sleep(0.3)
    print(json.dumps({'role': 'client', 'addr': sys.argv[2], 'n': int(sys.argv[4]), 'size': int(sys.argv[5]), 'lost': 0,
                      'rtt_us': {'p50': 40.0, 'p90': 45.0, 'p99': 60.0, 'max': 90.0}, 'host': host}))
