#!/usr/bin/env python3
"""Tables of a pathmap-session.sh run (OUT).

(a) Per Mac and controller, the stream's and thunderbolt_net's path ends from the debugfs path tables: the NHI ring,
    and per path hop its initial credits, flow control (ingress/egress), drop, priority/weight, with the lane and NHI
    adapters' buffers (ADP_CS_4: NFC and total) and the lane state; with the devices closed, open and closed again.
    A Tx path is one hop in the NHI adapter's table (in HopID = the Tx ring, out = lane adapter, next HopID = the
    other host's in HopID); an Rx path is one hop in the lane adapter's table (in HopID from the other host, out = NHI
    adapter, next HopID = the Rx ring).
(b) Per cable and direction, the lone-frame hold over the fresh opens (pos: the fillers each lone frame needed into the
    other end, 8 = not answered; pp: fillers per round trip), with opens lost to ENXIO and other failures counted apart.
Then the hold next to the path settings each direction went through, and which settings split the directions that held
from those that did not.

usage: pathmap.py OUT                 the whole map (markdown)
       pathmap.py --cable OUT A-B     one line: that cable's holds and lost opens
       pathmap.py --held OUT          one line: what the held-open devices saw
"""
import collections, glob, gzip, json, os, re, sys

MACS = ['incite', 'inflight', 'alight', 'indite', 'invite', 'enlight']
TYPES = {0: 'inactive', 1: 'lane', 2: 'nhi', 0x0e0101: 'dp-in', 0x0e0102: 'dp-out', 0x100101: 'pcie-down',
         0x100102: 'pcie-up'}


def hop(d0, d1):
    d1 = d1 or 0
    return {'enable': d0 >> 31 & 1, 'next': d0 & 0x7ff, 'out': d0 >> 11 & 0x3f, 'credits': d0 >> 17 & 0x7f,
            'pmps': d0 >> 24 & 1, 'weight': d1 & 0xf, 'prio': d1 >> 8 & 7, 'drop': d1 >> 11 & 1,
            'counter': d1 >> 12 & 0x7ff, 'counter_en': d1 >> 23 & 1, 'ifc': d1 >> 24 & 1, 'efc': d1 >> 25 & 1,
            'ishared': d1 >> 26 & 1, 'eshared': d1 >> 27 & 1, 'pending': d1 >> 28 & 1, 'raw': (d0, d1)}


def hop_text(h):
    if h is None:
        return '-'
    s = (f"cr {h['credits']} fc {h['ifc']}/{h['efc']} drop {h['drop']} p{h['prio']}/w{h['weight']}"
         f"{' pmps' if h['pmps'] else ''}{' shared ' + str(h['ishared']) + '/' + str(h['eshared']) if h['ishared'] or h['eshared'] else ''}"
         f"{' pending' if h['pending'] else ''}{' ctr ' + str(h['counter']) if h['counter_en'] else ''}")
    return s if h['enable'] else f'off ({s})'


def parse_dump(path):
    d = {'host': None, 'kernel': None, 'links': [], 'groups': {}, 'log': [], 'ports': {}}
    if not os.path.exists(path):
        path += '.gz'
        if not os.path.exists(path):
            return d
    f = gzip.open(path, 'rt', errors='replace') if path.endswith('.gz') else open(path, errors='replace')
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('# host '):
            f = line.split()
            d['host'], d['kernel'] = f[2], f[3]
        elif line.startswith('# links '):
            f = line[8:].split('\t')
            if len(f) >= 10 and f[0] != 'xdomain':
                d['links'].append(f)
        elif line.startswith('# group '):
            f = line[8:].split()
            m = re.search(r'/stream/(\d+-\d+)\.\d+/', f[0])
            if m:
                d['groups'][m.group(1)] = dict(kv.split('=', 1) for kv in f[1:] if '=' in kv)
        elif line.startswith('# kernel '):
            d['log'].append(line[9:])
        elif not line.startswith('#'):
            f = line.split()
            if len(f) < 7 or f[1] == '-':
                continue
            port = d['ports'].setdefault((f[0], int(f[1][4:])), {'regs': {}, 'path': {}})
            try:
                if f[2] == 'path' and len(f) == 7:
                    port['path'].setdefault(int(f[5], 16), [0, 0])[int(f[4])] = int(f[6], 16)
                elif f[2] == 'regs' and len(f) == 8:
                    port['regs'][(int(f[5], 16), int(f[6], 16), int(f[4]))] = int(f[7], 16)
            except ValueError:
                pass
    return d


def adapter(d, router, port):
    p = d['ports'].get((router, port), {'regs': {}})
    r = p['regs']
    out = {'type': TYPES.get(r.get((0, 0, 2), -1) & 0xffffff, '?') if (0, 0, 2) in r else '?'}
    if (0, 0, 4) in r:
        v = r[(0, 0, 4)]
        out.update(nfc=v & 0x3ff, total=v >> 20 & 0x3ff, lck=v >> 31)
    if (1, 0, 1) in r:
        v = r[(1, 0, 1)]
        out.update(lane_cs1=v, speed=v >> 16 & 0xf, width=v >> 20 & 0x3f, lb=v >> 15 & 1)
    return out


def adapter_text(a):
    if 'nfc' not in a:
        return '-'
    s = f"nfc {a['nfc']} total {a['total']}"
    if 'width' in a:
        s += f" width {a['width']} speed {a['speed']:#x}{' lb' if a['lb'] else ''}"
    return s


def ends(d):
    """Per link (xdomain), the path ends: stream and thunderbolt_net Tx and Rx hops, and the adapters."""
    out = {}
    for l in d['links']:
        xd, peer, dev = l[0], l[1], l[9]
        dom, lane = xd.split('-')
        router, lane = f'{dom}-0', int(lane)
        nhi = next((p for (r, p) in d['ports'] if r == router and adapter(d, r, p)['type'] == 'nhi'), None)
        g = d['groups'].get(xd, {})
        s_in, s_out = int(g.get('in_hopid', -1)), int(g.get('out_hopid', -1))
        e = {'xd': xd, 'peer': peer, 'dev': dev, 'state': l[5], 'lanes': l[4], 'router': router, 'lane': lane,
             'nhi': nhi, 'group': g, 'lane_adp': adapter(d, router, lane), 'lane1_adp': adapter(d, router, lane + 1),
             'nhi_adp': adapter(d, router, nhi) if nhi is not None else {}}
        lp = d['ports'].get((router, lane), {'path': {}})['path']
        np_ = d['ports'].get((router, nhi), {'path': {}})['path'] if nhi is not None else {}
        rx = {i: hop(*v) for i, v in lp.items() if i > 0 and v[0]}
        tx = {i: hop(*v) for i, v in np_.items() if i > 0 and v[0]}
        e['stream_rx'] = (s_in, rx.get(s_in))
        cand = [(i, h) for i, h in sorted(tx.items()) if h['out'] == lane and h['next'] == s_out]
        e['stream_tx'] = next(((i, h) for i, h in cand if h['enable']), cand[0] if cand else (None, None))
        e['stream_tx_all'] = cand
        e['tbnet_rx'] = [(i, h) for i, h in sorted(rx.items()) if h['enable'] and i != s_in and h['out'] == nhi]
        e['tbnet_tx'] = [(i, h) for i, h in sorted(tx.items()) if h['enable'] and h['out'] == lane and h['next'] != s_out]
        out[peer] = e
    return out


def load(outdir):
    snaps = {}
    for name in ('closed', 'open', 'after'):
        for h in MACS:
            p = os.path.join(outdir, f'paths-{name}-{h}.txt')
            if os.path.exists(p) or os.path.exists(p + '.gz'):
                snaps.setdefault(name, {})[h] = parse_dump(p)
    return snaps


def err_sections(path):
    """{(run, side): text} from an err file with '== run N side' headers."""
    out, key = {}, None
    if not os.path.exists(path):
        return out
    for line in open(path, errors='replace'):
        m = re.match(r'== run (\d+) (server|client)', line)
        if m:
            key = (int(m.group(1)), m.group(2))
            out[key] = ''
        elif key:
            out[key] += line
    return out


def why(text, rc, other_failed=False):
    """Why one end of a run failed ('' if it did not): ENXIO, busy, timeout (30 s, e.g. a server whose device read a
    stale CLOSE and so never saw the client), stopped (ended by the window after the other end failed), or the error."""
    if 'Errno 6]' in text or 'No such device or address' in text or 'ENXIO' in text:
        return 'ENXIO'
    if 'Errno 11]' in text or 'Errno 16]' in text or 'temporarily unavailable' in text:
        return 'busy'
    if rc == 124:
        return 'timeout'
    if rc in (137, 143, 130) or (rc == 255 and other_failed and not text.strip()):
        return 'stopped'
    last = [l for l in text.splitlines() if l.strip()]
    if last:
        return 'error: ' + last[-1][:80]
    return f'rc {rc}' if rc else ''


def cable_runs(outdir, name):
    """Per (kind, client): a list of opens, each {'run', 'hold' (text or None), 'lost' (why or None), ...}."""
    a, b = name.split('-')
    runs = {}
    rp = os.path.join(outdir, 'runs.txt')
    for line in open(rp) if os.path.exists(rp) else []:
        f = line.split()
        if len(f) >= 10 and f[1] == name:
            runs[int(f[0])] = {'kind': f[2], 'client': f[5], 'rc': int(f[6]), 'server': f[8], 'rs': int(f[9])}
    lines = {}
    for kind in ('pos', 'pp'):
        p = os.path.join(outdir, f'{kind}-{name}.jsonl')
        for l in open(p) if os.path.exists(p) else []:
            if l.startswith('{'):
                v = json.loads(l)
                lines[(v.get('run'), v.get('side'))] = v
    errs = {}
    for kind in ('pos', 'pp'):
        for h in (a, b):
            for k, t in err_sections(os.path.join(outdir, 'err', f'{kind}-{name}-{h}.txt')).items():
                errs[(kind, h) + k] = t
    out = collections.OrderedDict()
    for run in sorted(runs):
        r = runs[run]
        kind, c, s = r['kind'], r['client'], r['server']
        cl, sv = lines.get((run, 'client')), lines.get((run, 'server'))
        ce = why(errs.get((kind, c, run, 'client'), ''), r['rc'])
        if not ce and cl is not None and not cl.get('rounds'):
            ce = 'no answer'
        se = why(errs.get((kind, s, run, 'server'), ''), r['rs'], other_failed=bool(ce))
        hold = lost = None
        if kind == 'pos' and cl and cl.get('raw'):
            ex = sorted(set(x[2] for x in cl['raw']))
            hold = '/'.join(str(x) for x in ex)
        elif kind == 'pp' and cl and cl.get('rounds'):
            f = cl.get('fillers', {})
            hold = str(f.get('p50')) + ('' if f.get('max') == f.get('p50') else f"-{f.get('max')}")
        if hold is None:
            parts = [f'{w} at {h} ({side})' for w, h, side in ((ce, c, 'client'), (se, s, 'server')) if w]
            lost = ', '.join(parts) or ('no rounds' if cl else 'no client line')
        out.setdefault((kind, c), []).append({'run': run, 'hold': hold, 'lost': lost, 'client': cl, 'server': sv,
                                              'client_why': ce, 'server_why': se})
    return out


def cable_line(outdir, name):
    a, b = name.split('-')
    runs = cable_runs(outdir, name)
    parts = []
    for kind in ('pos', 'pp'):
        for c, s in ((a, b), (b, a)):
            opens = runs.get((kind, c), [])
            arrow = f'{c}->{s}' if kind == 'pos' else f'client {c}'
            parts.append(f"{kind} {arrow} [{' '.join(o['hold'] or 'x' for o in opens)}]")
    lost = collections.Counter(o['lost'] for v in runs.values() for o in v if o['lost'])
    s = f"{name}: " + '; '.join(parts)
    if lost:
        s += '; lost opens: ' + ', '.join(f'{n}x {w}' for w, n in lost.items())
    return s


def held_line(outdir):
    out = []
    for h in MACS:
        p = os.path.join(outdir, f'holdopen-{h}.jsonl')
        for l in open(p) if os.path.exists(p) else []:
            if l.startswith('{'):
                v = json.loads(l)
                ds = []
                for d in v['devs']:
                    t = d['dev'].rsplit('/', 1)[-1] + (' ok' if d.get('opened') else ' ' + str(d.get('error')))
                    if d.get('eofs'):
                        t += f" eofs {d['eofs']}"
                    if d.get('errors'):
                        t += ' ' + ' '.join(f'{k} {n}' for k, n in d['errors'].items())
                    ds.append(t)
                out.append(f"{h}: {', '.join(ds)}")
        e = os.path.join(outdir, 'err', f'holdopen-{h}.txt')
        if os.path.exists(e) and os.path.getsize(e):
            out.append(f'{h} errors: ' + open(e, errors='replace').read().strip().splitlines()[-1][:100])
    return '; '.join(out)


def ring_text(i, h):
    return '-' if h is None else f'{i}'


def main():
    if sys.argv[1] == '--cable':
        print(cable_line(sys.argv[2], sys.argv[3]))
        return
    if sys.argv[1] == '--held':
        print(held_line(sys.argv[2]))
        return
    outdir = sys.argv[1]
    snaps = load(outdir)
    print(f'# Path map: {os.path.basename(outdir.rstrip("/"))}\n')
    # (a) per Mac and controller, devices open.
    print('## Path ends with the stream devices open\n')
    print('Tx: the hop in the NHI adapter\'s table (in HopID = Tx ring); Rx: the hop in the lane adapter\'s table '
          '(next HopID = Rx ring). cr = initial credits, fc = ingress/egress flow control.\n')
    print('| Mac | kernel | link | peer | stream Tx ring>hop | stream Tx hop | stream Rx hop>ring | stream Rx hop '
          '| tbnet Tx ring>hop | tbnet Tx hop | tbnet Rx hop>ring | tbnet Rx hop | lane adapter | NHI adapter |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    allends = {}
    for name in ('closed', 'open', 'after'):
        for h, d in snaps.get(name, {}).items():
            allends[(name, h)] = ends(d)
    for h in MACS:
        d = snaps.get('open', {}).get(h)
        if not d:
            continue
        for peer, e in allends[('open', h)].items():
            ti, th = e['stream_tx']
            ri, rh = e['stream_rx']
            tb_tx = '; '.join(f"{i}>{x['next']}" for i, x in e['tbnet_tx']) or '-'
            tb_txh = '; '.join(hop_text(x) for i, x in e['tbnet_tx']) or '-'
            tb_rx = '; '.join(f"{i}>{x['next']}" for i, x in e['tbnet_rx']) or '-'
            tb_rxh = '; '.join(hop_text(x) for i, x in e['tbnet_rx']) or '-'
            print(f"| {h} | {d['kernel']} | {e['xd']} port {e['lane']} | {peer} | "
                  f"{ti if th else '-'}>{th['next'] if th else '-'} | {hop_text(th)} | "
                  f"{ri}>{rh['next'] if rh else '-'} | {hop_text(rh)} | {tb_tx} | {tb_txh} | {tb_rx} | {tb_rxh} | "
                  f"port {e['lane']}: {adapter_text(e['lane_adp'])}; port {e['lane'] + 1}: {adapter_text(e['lane1_adp'])} | "
                  f"port {e['nhi']}: {adapter_text(e['nhi_adp'])} |")
    # What changed between the snapshots.
    print('\n## Changes between the snapshots\n')
    changes = expected = 0
    # A stream hop that is only switched on (open) or off (closed again), its flow control set or cleared with it,
    # is expected; anything else is listed.
    fc_enable = (1 << 31, 1 << 24 | 1 << 25 | 1 << 26 | 1 << 27)
    for h in MACS:
        for a, b in (('closed', 'open'), ('open', 'after')):
            da, db = snaps.get(a, {}).get(h), snaps.get(b, {}).get(h)
            if not da or not db:
                continue
            stream_hops = set()
            for e in ends(db).values():
                stream_hops.add((e['router'], e['lane'], e['stream_rx'][0]))
                stream_hops.update((e['router'], e['nhi'], i) for i, _ in e['stream_tx_all'])
            for key in sorted(set(da['ports']) | set(db['ports'])):
                pa, pb = da['ports'].get(key, {}).get('path', {}), db['ports'].get(key, {}).get('path', {})
                for i in sorted(set(pa) | set(pb)):
                    if pa.get(i) != pb.get(i):
                        va, vb = pa.get(i), pb.get(i)
                        if (va and vb and key + (i,) in stream_hops and
                                all(x & ~m == y & ~m for x, y, m in zip(va, vb, fc_enable))):
                            expected += 1
                            continue
                        changes += 1
                        print(f"- {h} {key[0]} port {key[1]} in HopID {i}, {a} -> {b}: "
                              f"{' '.join(f'{x:#010x}' for x in va) if va else '-'} -> "
                              f"{' '.join(f'{x:#010x}' for x in vb) if vb else '-'}")
                ra, rb = da['ports'].get(key, {}).get('regs', {}), db['ports'].get(key, {}).get('regs', {})
                for k in sorted(set(ra) | set(rb)):
                    if k[:2] == (0, 0) and k[2] == 4 and ra.get(k) != rb.get(k):
                        changes += 1
                        print(f"- {h} {key[0]} port {key[1]} ADP_CS_4, {a} -> {b}: {ra.get(k)} -> {rb.get(k)}")
    print(f'- {changes} other changes; {expected} stream hops only switched on or off with the devices '
          f'(enable and flow control bits)')
    # (b) the hold per direction next to the path settings it went through.
    print('\n## The hold per direction (pos: fillers each lone frame needed into the other end, per open)\n')
    print('| direction | pos per open | pp per open (client = sender) | lost opens | sender Tx ring>hop | sender Tx hop '
          '| receiver Rx hop>ring | receiver Rx hop | tbnet rings (sender Tx, receiver Rx) | receiver lane adapter '
          '| sender lane adapter | kernels | receiver group |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    rows = []
    cables = sorted({os.path.basename(p)[4:-6] for p in glob.glob(os.path.join(outdir, 'pos-*.jsonl'))} |
                    {os.path.basename(p)[3:-6] for p in glob.glob(os.path.join(outdir, 'pp-*.jsonl'))},
                    key=lambda n: (MACS.index(n.split('-')[0]) if n.split('-')[0] in MACS else 9, n))
    for name in cables:
        runs = cable_runs(outdir, name)
        a, b = name.split('-')
        for x, y in ((a, b), (b, a)):
            pos = runs.get(('pos', x), [])
            pp = runs.get(('pp', x), [])
            lost = [f"run {o['run']} {o['lost']}" for o in pos + pp if o['lost']]
            ex = allends.get(('open', x), {}).get(y)
            ey = allends.get(('open', y), {}).get(x)
            row = {'direction': f'{x}->{y}', 'pos': [o['hold'] for o in pos], 'pp': [o['hold'] for o in pp]}
            if ex and ey:
                ti, th = ex['stream_tx']
                ri, rh = ey['stream_rx']
                row.update({
                    'tx_ring': ti, 'tx_credits': th and th['credits'], 'tx_fc': th and f"{th['ifc']}/{th['efc']}",
                    'tx_drop': th and th['drop'], 'tx_prio': th and f"{th['prio']}/{th['weight']}",
                    'rx_ring': rh and rh['next'], 'rx_credits': rh and rh['credits'],
                    'rx_fc': rh and f"{rh['ifc']}/{rh['efc']}", 'rx_drop': rh and rh['drop'],
                    'tbnet_tx_ring': ','.join(str(i) for i, _ in ex['tbnet_tx']),
                    'tbnet_rx_ring': ','.join(str(h['next']) for _, h in ey['tbnet_rx']),
                    'stream_tx_before_tbnet': (ti is not None and bool(ex['tbnet_tx']) and ti < ex['tbnet_tx'][0][0]),
                    'stream_rx_before_tbnet': (rh is not None and bool(ey['tbnet_rx']) and rh['next'] < ey['tbnet_rx'][0][1]['next']),
                    'rx_lane_buffers': adapter_text(ey['lane_adp']), 'tx_lane_buffers': adapter_text(ex['lane_adp']),
                    'rx_nhi_buffers': adapter_text(ey['nhi_adp']), 'tx_nhi_buffers': adapter_text(ex['nhi_adp']),
                    'sender_kernel': snaps['open'][x]['kernel'], 'receiver_kernel': snaps['open'][y]['kernel'],
                    'sender_ctl': ex['router'], 'receiver_ctl': ey['router'],
                    'receiver_group': ' '.join(f'{k}={v}' for k, v in sorted(ey['group'].items())
                                               if k not in ('in_hopid', 'out_hopid', 'index'))})
                print(f"| {x}->{y} | {' '.join(o['hold'] or 'x' for o in pos)} | {' '.join(o['hold'] or 'x' for o in pp)} "
                      f"| {'; '.join(lost) or '-'} | {ti}>{th['next'] if th else '-'} | {hop_text(th)} "
                      f"| {ri}>{rh['next'] if rh else '-'} | {hop_text(rh)} | {row['tbnet_tx_ring']}, {row['tbnet_rx_ring']} "
                      f"| {row['rx_lane_buffers']} | {row['tx_lane_buffers']} | {row['sender_kernel']} > {row['receiver_kernel']} "
                      f"| {row['receiver_group'] or '-'} |")
            else:
                print(f"| {x}->{y} | {' '.join(o['hold'] or 'x' for o in pos)} | {' '.join(o['hold'] or 'x' for o in pp)} "
                      f"| {'; '.join(lost) or '-'} | no path table | | | | | | | | |")
            rows.append(row)
    # Which settings split the directions that held from the others.
    print('\n## Settings against the hold\n')

    def held(r):
        v = [h for h in r['pos'] if h is not None]
        if not v:
            return None
        return any(h != '0' for h in v)

    yes = [r for r in rows if held(r) is True]
    no = [r for r in rows if held(r) is False]
    print(f"Held (pos, any open with a filler): {', '.join(r['direction'] for r in yes) or 'none'}. "
          f"No hold: {', '.join(r['direction'] for r in no) or 'none'}. "
          f"No answer: {', '.join(r['direction'] for r in rows if held(r) is None) or 'none'}.\n")
    keys = [k for k in (rows[0].keys() if rows else []) if k not in ('direction', 'pos', 'pp')]
    keys = sorted({k for r in rows for k in r} - {'direction', 'pos', 'pp'})
    print('| setting | in the directions that held | in the others | splits them |')
    print('|---|---|---|---|')
    for k in keys:
        vy = sorted({str(r.get(k)) for r in yes})
        vn = sorted({str(r.get(k)) for r in no})
        split = bool(yes) and bool(no) and not set(vy) & set(vn)
        print(f"| {k} | {', '.join(vy) or '-'} | {', '.join(vn) or '-'} | {'yes' if split else ''} |")
    # The held-open devices and the kernel log lines.
    print('\n## Devices held open for the open snapshot\n')
    print(held_line(outdir) or '-')
    print('\n## Kernel log (last Thunderbolt lines per Mac, closed snapshot)\n')
    for h in MACS:
        d = snaps.get('closed', {}).get(h)
        if d and d['log']:
            print(f'{h}:')
            for l in d['log'][-12:]:
                print(f'    {l}')


if __name__ == '__main__':
    main()
