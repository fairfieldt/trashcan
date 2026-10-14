#!/usr/bin/env python3
"""Transport of every link in a six-Mac session's runs, from the driver's and the nodes' logs, per process and peer,
in order: "stream up" (an epoch opened), "stream epoch ... not opened" (a failed attempt; the next epoch may open),
"stream down ...; continuing over TCP", "no stream". A link is stream (up and never down), tcp (never up), or
stream-then-tcp (down after up, whatever came later). Then whether the two arms of each A/B pair ran every link the
same way (a pair where they did not is not a valid comparison).
usage: transport.py SESSION_DIR TAG [--check RUN]   (driver logs TAG-RUN.err in SESSION_DIR, node logs
HOST/node-TAG-RUN-HOST.log; --check: exit status 0 if RUN ran all 10 link-ends on their streams, else 1)"""
import glob, os, re, sys

d, tag = sys.argv[1], sys.argv[2]
check = sys.argv[4] if sys.argv[3:4] == ['--check'] else None
LINE = re.compile(r'^link (\S+) \(([^)]*)\): (stream up|stream down|stream epoch \S+ not opened|no stream)')


def links(path):
    state = {}
    for line in open(path, errors='replace'):
        m = LINE.match(line.strip())
        if not m:
            continue
        peer, what = m.group(1), m.group(3)
        st = state.get(peer, 'tcp')
        if what == 'stream up':
            state[peer] = st if st == 'stream-then-tcp' else 'stream'
        elif what.startswith('stream down'):
            state[peer] = 'stream-then-tcp'
        else:
            state[peer] = st
    return state


runs = {}
for f in glob.glob(os.path.join(d, f'{tag}-*.err')):
    run = os.path.basename(f)[len(tag) + 1:-4]
    runs.setdefault(run, {})['driver'] = links(f)
for f in glob.glob(os.path.join(d, '*', f'node-{tag}-*.log')):
    host = os.path.basename(os.path.dirname(f))
    run = os.path.basename(f)[len(f'node-{tag}-'):-len(f'-{host}.log')]
    runs.setdefault(run, {})[host] = links(f)


def sig(run):
    return tuple(sorted((proc, peer, st) for proc, ls in runs[run].items() for peer, st in ls.items()))


if check is not None:
    s = sig(check) if check in runs else ()
    ok = len(s) >= 10 and all(st == 'stream' for _, _, st in s)
    print(f"{check}: {sum(1 for x in s if x[2] == 'stream')} of {len(s)} link-ends on their stream")
    sys.exit(0 if ok else 1)
for run in sorted(runs):
    s = sig(run)
    other = [f'{p}->{q} {st}' for p, q, st in s if st != 'stream']
    print(f"{run:18s} {sum(1 for x in s if x[2] == 'stream'):2d} stream{'; ' + ', '.join(other) if other else ''}")
print()
for run in sorted(runs):
    m = re.match(r'(.*)-base(-\d+)?$', run)
    if not m:
        continue
    other = f"{m.group(1)}-cand{m.group(2) or ''}"
    if other in runs:
        same = sig(run) == sig(other)
        print(f"{m.group(1)}{m.group(2) or ''}: {'same transport' if same else 'DIFFERENT transport: invalid pair'}")
