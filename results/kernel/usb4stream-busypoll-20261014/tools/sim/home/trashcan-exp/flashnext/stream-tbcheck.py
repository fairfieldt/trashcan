#!/usr/bin/env python3
"""Stub of tbcheck.py on a simulated Mac (pp and pos). The ring is incite -> inflight -> alight -> indite -> invite ->
enlight -> incite (tbstream0 to the next Mac, tbstream1 to the one before). Lone frames from incite to inflight wait
for 2 fillers and between alight and indite for 3; elsewhere none. On the stock Macs a run now and then fails as on
stock 7.3: the client (run % 11 == 5) or the server (run % 13 == 7) gets ENXIO; the server of a failed client waits
for it until it is ended, as the real one does."""
import json, os, sys, time

RING = ['incite', 'inflight', 'alight', 'indite', 'invite', 'enlight']
STOCK = {'incite', 'inflight', 'alight', 'indite'}
HOLD = {('incite', 'inflight'): 2, ('alight', 'indite'): 3, ('indite', 'alight'): 3}
host = os.environ.get('SIMHOST', '?')
step, dev, side = sys.argv[1], sys.argv[2], sys.argv[3]
run = int(sys.argv[-1])
i = RING.index(host)
other = RING[(i + 1) % 6] if dev.endswith('0') else RING[i - 1]
client, server = (host, other) if side == 'client' else (other, host)
enxio = 'Traceback (most recent call last):\n  File "stream-tbcheck.py", line 120, in write\nOSError: [Errno 6] No such device or address'
client_fails = run % 11 == 5 and client in STOCK
server_fails = run % 13 == 7 and server in STOCK and not client_fails
# A server whose device read a stale CLOSE never hears the client: the client gives up with no rounds.
server_deaf = run % 17 == 3 and server in STOCK and not client_fails and not server_fails
if side == 'client' and client_fails:
    time.sleep(0.3)
    print(enxio, file=sys.stderr)
    sys.exit(1)
if side == 'server' and (client_fails or server_deaf):
    time.sleep(30)
    sys.exit(0)
if side == 'client' and server_deaf:
    time.sleep(1)
    print(json.dumps({'step': step, 'side': 'client', 'dev': dev, 'run': run, 'rounds': 0, 'raw': [], 'host': host}))
    sys.exit(0)
if side == 'server' and server_fails:
    time.sleep(0.2)
    print(enxio, file=sys.stderr)
    sys.exit(1)
time.sleep(0.5 if side == 'server' else 0.3)
hold = HOLD.get((client, server), 0)
if step == 'pp':
    n = 0 if server_fails else int(sys.argv[4])
    f = max(hold, HOLD.get((server, client), 0))
    print(json.dumps({'step': 'pp', 'side': side, 'dev': dev, 'size': 112, 'follow': 0, 'run': run, 'rounds': n,
                      'us': {'n': n, 'p50': 20000.0 * f + 60.0}, 'fillers': {'n': n, 'p50': f, 'max': f} if n else {},
                      'host': host}))
elif side == 'client':
    rounds = 0 if server_fails else int(sys.argv[4])
    print(json.dumps({'step': 'pos', 'side': 'client', 'dev': dev, 'run': run, 'rounds': rounds,
                      'raw': [[4 * r, r % 4, hold, 20000 * hold + 50] for r in range(rounds)], 'host': host}))
else:
    print(json.dumps({'step': 'pos', 'side': 'server', 'dev': dev, 'run': run, 'answered': int(sys.argv[4]),
                      'host': host}))
