#!/usr/bin/env python3
"""Transport of every link in a six-Mac session's runs, from the driver's and the nodes' logs, per process and peer,
in order: "stream up" (an epoch opened), "stream epoch ... not opened" (a failed attempt; the next epoch may open),
"stream down ...", "no stream". A link-end is
  stream           up and never down;
  closed at end    the driver's (or the previous node's) close at its exit, seen by the next node after the run: the
                   node's last line for its upstream link is the only down or failure on it, "the other side closed its
                   stream", after an up; the upstream end logged an up and no down, failed epoch or "not opened"; the
                   run's rc (TAG-RUN.rc) is 0. Counted as on its stream, and printed;
  tcp              never up;
  stream-then-tcp  any other down.
Then whether the two arms of each A/B pair ran every link the same way (a pair where they did not is not a valid
comparison). The chain order (driver first) is read from SESSION_DIR/chain.txt.
usage: transport.py SESSION_DIR TAG [--check RUN]   (driver logs TAG-RUN.err in SESSION_DIR, node logs
HOST/node-TAG-RUN-HOST.log; --check: exit status 0 if RUN ran all 10 link-ends on their streams, else 1, printing
each link-end not plainly on its stream with its log lines)"""
import glob, os, re, sys

d, tag = sys.argv[1], sys.argv[2]
check = sys.argv[4] if sys.argv[3:4] == ['--check'] else None
LINE = re.compile(r'^link (\S+) \(([^)]*)\): (stream up|stream down|stream epoch \S+ not opened|no stream)(.*)$')
CLOSE = 'stream down (the other side closed its stream); continuing over TCP'
f = os.path.join(d, 'chain.txt')
chain = open(f).read().split()[0].split(',') if os.path.exists(f) else []


def events(path):
    """peer -> [(kind, text)], kind up / down / failed (not opened, no stream)"""
    ev = {}
    for line in open(path, errors='replace'):
        m = LINE.match(line.strip())
        if m:
            w = m.group(3)
            kind = 'up' if w == 'stream up' else 'down' if w == 'stream down' else 'failed'
            ev.setdefault(m.group(1), []).append((kind, w + m.group(4)))
    return ev


def state(evs):
    st = 'tcp'
    for kind, _ in evs:
        if kind == 'up':
            st = st if st == 'stream-then-tcp' else 'stream'
        elif kind == 'down':
            st = 'stream-then-tcp'
    return st


runs, paths = {}, {}
for f in glob.glob(os.path.join(d, f'{tag}-*.err')):
    run = os.path.basename(f)[len(tag) + 1:-4]
    runs.setdefault(run, {})['driver'] = events(f)
    paths[(run, 'driver')] = f
for f in glob.glob(os.path.join(d, '*', f'node-{tag}-*.log')):
    host = os.path.basename(os.path.dirname(f))
    run = os.path.basename(f)[len(f'node-{tag}-'):-len(f'-{host}.log')]
    runs.setdefault(run, {})[host] = events(f)
    paths[(run, host)] = f


def rc(run):
    try:
        return int(open(os.path.join(d, f'{tag}-{run}.rc')).read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def closed_at_end(run, proc, peer, evs):
    # proc is a node (not the driver) and peer is its upstream: the driver or the previous node
    if proc == 'driver' or proc not in chain or chain.index(proc) == 0 or chain[chain.index(proc) - 1] != peer:
        return False
    bad = [e for e in evs if e[0] != 'up']
    if bad != [('down', CLOSE)] or evs[-1] != bad[0] or evs[0][0] != 'up' or rc(run) != 0:
        return False
    other = runs[run].get('driver' if peer == chain[0] else peer, {}).get(proc, [])
    return bool(other) and all(kind == 'up' for kind, _ in other)


def classify(run):
    out = []
    for proc, ev in runs[run].items():
        for peer, evs in ev.items():
            st = 'closed at end' if closed_at_end(run, proc, peer, evs) else state(evs)
            out.append((proc, peer, st))
    return sorted(out)


def sig(run):  # the transport each link-end ran on
    return tuple((p, q, 'stream' if st == 'closed at end' else st) for p, q, st in classify(run))


def show(run, proc, peer, st):
    print(f"  {proc} -> {peer}: {st} ({paths[(run, proc)]})")
    for line in open(paths[(run, proc)], errors='replace'):
        if line.startswith(f'link {peer} '):
            print(f"    {line.rstrip()}")


if check is not None:
    s = classify(check) if check in runs else []
    ok = len(s) >= 10 and all(st in ('stream', 'closed at end') for _, _, st in s)
    n, c = sum(1 for x in s if x[2] in ('stream', 'closed at end')), sum(1 for x in s if x[2] == 'closed at end')
    print(f"{check}: {n} of {len(s)} link-ends on their stream{f' ({c} closed at end)' if c else ''}")
    for proc, peer, st in s:
        if st != 'stream':
            show(check, proc, peer, st)
    sys.exit(0 if ok else 1)
for run in sorted(runs):
    s = classify(run)
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
