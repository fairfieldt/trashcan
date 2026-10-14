#!/usr/bin/env python3
"""Account for a decode step of a traced chain run (TRASHCAN_Q4_TRACE on every process; engine/src/qwen4/trace.rs).

Every machine's events are moved to the driver's clock with the offsets the driver measured by ping-pong before and
after the decode steps (linear between them). Each decode step is the window from the driver's `fwd` mark to the next
one; the critical path is followed through the chain: the driver's host work and stages, each hop, each machine's
pickup, stages and send, the final hidden state's way back, the head scoring and the candidates.

usage: timeline.py DRIVER.json NODE.json... (nodes in chain order; .json.gz too) [--skip N] [--json OUT]
"""
import argparse, gzip, json, statistics

p = argparse.ArgumentParser()
p.add_argument('driver')
p.add_argument('nodes', nargs='*')
p.add_argument('--skip', type=int, default=2, help='decode steps to skip at the start')
p.add_argument('--json', help='write the per-step segments here')
p.add_argument('--steps', action='store_true', help='print every step')
a = p.parse_args()

def load(path):
    return json.load(gzip.open(path, 'rt') if path.endswith('.gz') else open(path))


drv = load(a.driver)
nodes = [load(n) for n in a.nodes]
clocks = drv['extra']['clocks']
assert len(clocks) >= 1 and all(len(c['offsets']) == len(nodes) for c in clocks), 'offsets do not match the nodes'


def to_driver(i, t):
    """Node i's local time t on the driver's clock."""
    if len(clocks) == 1:
        return t - clocks[0]['offsets'][i][0]
    c0, c1 = clocks[0], clocks[-1]
    o0, o1 = c0['offsets'][i][0], c1['offsets'][i][0]
    td = t - o0
    return t - (o0 + (o1 - o0) * (td - c0['t']) / (c1['t'] - c0['t']))


machines = [('driver', [(n, x, t) for n, x, t in drv['events']])]
for i, n in enumerate(nodes):
    machines.append((n['host'], sorted(((e, x, to_driver(i, t)) for e, x, t in n['events']), key=lambda v: v[2])))
names = [drv['host']] + [n['host'] for n in nodes]

# Decode steps: driver `fwd` marks between the clock syncs.
t0 = clocks[0]['t']
t1 = clocks[-1]['t'] if len(clocks) > 1 else float('inf')
starts = [(x, t) for e, x, t in machines[0][1] if e == 'fwd' and t0 <= t <= t1]
dones = {x: t for e, x, t in machines[0][1] if e == 'done'}
steps = []
for k, (pos, ts) in enumerate(starts):
    te = starts[k + 1][1] if k + 1 < len(starts) else dones.get(pos, ts) + 1e6
    steps.append((pos, ts, te))
steps = steps[a.skip:]

W = 2560 * 4 * 4  # residual bytes per token
H = 2560 * 4      # final hidden state bytes


def window(evs, ts, te, lead=0):
    return [(e, x, t) for e, x, t in evs if ts - lead <= t < te]


def first(evs, name, arg=None, after=-1e30):
    for e, x, t in evs:
        if e == name and (arg is None or x == arg) and t >= after:
            return t
    return None


def stage_args(evs):
    seen = []
    for e, x, t in evs:
        if e == 'sub0' and x not in seen:
            seen.append(x)
    return seen


rows = []
for pos, ts, te in steps:
    seg = {}  # name -> ms on the critical path
    info = {}
    ms = lambda d: d / 1e6
    # Driver: embed and n-gram rows, then its stages.
    ev = window(machines[0][1], ts, te)
    t_fwd = first(ev, 'fwd')
    t_ple = first(ev, 'ple')
    seg['driver: embed + n-gram rows'] = ms(t_ple - t_fwd)
    prev = t_ple
    chain_ok = True
    for m, (host, evs) in enumerate(machines):
        ev = window(evs, ts, te, lead=0)
        if m > 0:
            # Arrival: the reader thread has the whole frame, then the session thread takes it.
            rx0 = first(ev, 'rx_up0', W, after=prev - 1e6)
            rx1 = first(ev, 'rx_up1', W, after=prev - 1e6)
            deq = first(ev, 'fwd', pos)
            if None in (rx0, rx1, deq):
                chain_ok = False
                break
            seg[f'{host}: hop in (first bytes)'] = ms(rx0 - prev)
            seg[f'{host}: hop in (rest of the frame)'] = ms(rx1 - rx0)
            seg[f'{host}: reader thread -> session'] = ms(deq - rx1)
            prev = deq
        for s, arg in enumerate(stage_args(ev)):
            sub0, sub1 = first(ev, 'sub0', arg), first(ev, 'sub1', arg)
            g0, g1 = first(ev, 'gpu0', arg), first(ev, 'gpu1', arg)
            fence, col = first(ev, 'fence', arg), first(ev, 'col', arg)
            seg[f'{host}: before stage {arg} (copies)'] = ms(sub0 - prev)
            if g0 is not None:
                seg[f'{host}: stage {arg} submit -> GPU start'] = ms(g0 - sub0)
                seg[f'{host}: stage {arg} GPU'] = ms(g1 - g0)
                seg[f'{host}: stage {arg} GPU end -> fence wakeup'] = ms(fence - g1)
                info[f'{host}: stage {arg} submit call'] = ms(sub1 - sub0)
            else:
                seg[f'{host}: stage {arg} submit -> fence'] = ms(fence - sub0)
            seg[f'{host}: stage {arg} collect (stamps, output copy)'] = ms(col - fence)
            prev = col
        if m + 1 < len(machines):
            tx1 = first(ev, 'tx_dn1', W, after=prev)
            if tx1 is None:
                chain_ok = False
                break
            seg[f'{host}: send residual'] = ms(tx1 - prev)
            prev = tx1
    if not chain_ok:
        continue
    # The way back: h from the machine that ends the model up to the driver.
    last = len(machines) - 1
    host, evs = machines[last]
    ev = window(evs, ts, te)
    tx1 = first(ev, 'tx_up1', H, after=prev)
    seg[f'{host}: send h'] = ms(tx1 - prev)
    prev = tx1
    for m in range(last - 1, -1, -1):
        host, evs = machines[m]
        ev = window(evs, ts, te)
        rx1 = first(ev, 'rx_dn1', H, after=prev - 1e6)
        seg[f'{host}: h hop in'] = ms(rx1 - prev)
        prev = rx1
        if m > 0:
            tx1 = first(ev, 'tx_up1', H, after=prev)
            seg[f'{host}: relay h'] = ms(tx1 - prev)
            prev = tx1
    # The driver scores its rows, then waits for the candidates of the others.
    ev = window(machines[0][1], ts, te)
    t_top = first(ev, 'top', after=prev)
    t_res = first(ev, 'rx_dn1', 512, after=prev) or first(ev, 'rx_dn1', 0, after=prev)
    t_done = first(ev, 'done', pos)
    g = {n: [t for e, x, t in ev if e == n and t >= prev] for n in ('hgpu0', 'hgpu1', 'hgpu2', 'hgpu3', 'hfence')}
    if all(g.values()):
        # GPU timestamps of the head graphs (both GPUs score at once): the later GPU at each step.
        seg['driver: head: h in -> GPU start'] = ms(max(g['hgpu0']) - prev)
        seg['driver: head: logits (headvec)'] = ms(max(g['hgpu1']) - max(g['hgpu0']))
        seg['driver: head: chunk top-k'] = ms(max(g['hgpu2']) - max(g['hgpu1']))
        seg['driver: head: merge'] = ms(max(g['hgpu3']) - max(g['hgpu2']))
        seg['driver: head: GPU end -> candidates'] = ms(t_top - max(g['hgpu3']))
    else:
        seg['driver: own head rows (h in -> candidates)'] = ms(t_top - prev)
    seg['driver: waiting for the others\' candidates'] = ms(max(t_res - t_top, 0))
    seg['driver: merge'] = ms(t_done - max(t_res, t_top))
    seg['driver: bench loop until the next step'] = ms(te - t_done)
    # Off the critical path: each machine's head scoring and when its candidates left.
    for m, (host, evs) in enumerate(machines):
        ev = window(evs, ts, te)
        h_in = first(ev, 'hsub0')
        top = first(ev, 'top')
        if h_in and top:
            info[f'{host}: head rows h -> candidates'] = ms(top - h_in)
        if m > 0:
            # The candidate frame: 64 binary pairs (512 bytes) or, before, JSON with an empty blob.
            res_tx = first(ev, 'tx_up1', 512) or first(ev, 'tx_up1', 0)
            res_rx = first(ev, 'rx_dn1', 512) or first(ev, 'rx_dn1', 0)
            if res_tx and top:
                info[f'{host}: candidates sent after own scoring'] = ms(res_tx - top)
            if res_tx and res_rx:
                info[f'{host}: candidates sent after those from below'] = ms(res_tx - res_rx)
    total = ms(te - ts)
    rows.append((pos, total, seg, info))

if not rows:
    raise SystemExit('no complete steps')


def med(vals):
    return statistics.median(vals)


keys = list(rows[0][2].keys())
print(f'{len(rows)} decode steps (skipped {a.skip}), machines: {" -> ".join(names)}')
print(f'step: median {med([r[1] for r in rows]):.3f} ms, mean {statistics.mean([r[1] for r in rows]):.3f} ms')
print(f'clock offsets (ns, round trip): {clocks[0]["offsets"]} -> {clocks[-1]["offsets"]}')
groups = {}
print(f'\n{"segment":60s} {"median ms":>10s} {"mean ms":>10s}')
for k in keys:
    vals = [r[2].get(k, 0.) for r in rows]
    print(f'{k:60s} {med(vals):10.3f} {statistics.mean(vals):10.3f}')
total_mean = statistics.mean([r[1] for r in rows])
seg_mean = statistics.mean([sum(r[2].values()) for r in rows])
print(f'{"sum of segments (mean)":60s} {"":10s} {seg_mean:10.3f}  (step mean {total_mean:.3f})')


def classify(k):
    s = k.split(': ', 1)[1]
    if s.endswith(' GPU'):
        return 'GPU work of the layer stages'
    if 'submit -> GPU start' in s:
        return 'stage submission to GPU start'
    if 'GPU end -> fence' in s:
        return 'GPU end to fence wakeup'
    if 'collect' in s or 'before stage' in s:
        return 'host copies around stages'
    if 'hop in' in s and 'h hop' not in s:
        return 'residual hops (wire)'
    if 'reader thread' in s:
        return 'reader thread to session thread'
    if s == 'send residual':
        return 'sending the residual'
    if 'h hop' in s or 'send h' in s or 'relay h' in s:
        return 'final hidden state back to the driver'
    if s.startswith('head: logits') or s.startswith('head: chunk') or s == 'head: merge':
        return 'head scoring on the GPU (driver)'
    if 'head' in s or 'candidates' in s or s == 'merge':
        return 'head scoring and candidates, host side'
    return s


cats = {}
for r in rows:
    for k, v in r[2].items():
        c = classify(k)
        cats.setdefault(c, []).append(0.)
for c in cats:
    cats[c] = statistics.mean([sum(v for k, v in r[2].items() if classify(k) == c) for r in rows])
print(f'\n{"category":60s} {"mean ms":>10s}')
for c, v in sorted(cats.items(), key=lambda kv: -kv[1]):
    print(f'{c:60s} {v:10.3f}')
ikeys = sorted({k for r in rows for k in r[3]})
print(f'\n{"off the critical path":60s} {"median ms":>10s}')
for k in ikeys:
    vals = [r[3][k] for r in rows if k in r[3]]
    print(f'{k:60s} {med(vals):10.3f}')
if a.steps:
    for pos, total, seg, info in rows:
        print(pos, f'{total:.3f}', ' '.join(f'{v:.3f}' for v in seg.values()))
if a.json:
    json.dump([{'pos': r[0], 'ms': r[1], 'segments': r[2], 'info': r[3]} for r in rows], open(a.json, 'w'), indent=1)
