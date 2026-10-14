#!/usr/bin/env python3
"""Tables from a window's `trashcan link-test` results (TAG-HOST.jsonl files, one JSON line per test).

usage: summary.py DIR [suites|soak|faults|all]
"""
import glob
import json
import os
import sys

LINKS = ['incite', 'indite', 'inflight', 'alight', 'invite', 'enlight']
NEXT = dict(zip(LINKS, LINKS[1:] + LINKS[:1]))


def lines(path):
    out = []
    if os.path.exists(path):
        for l in open(path):
            l = l.strip()
            if l.startswith('{'):
                try:
                    out.append(json.loads(l))
                except json.JSONDecodeError:
                    pass
    return out


def results(d, tag, host):
    return [x for x in lines(f'{d}/{tag}-{host}.jsonl') if 'result' in x or 'error' in x]


def paths(r):
    return '+'.join(sorted({l['path'] for l in r.get('lanes', [])})) or '?'


def suites(d):
    print('| suite | link | path | 64 B ping-pong p50 / p99 µs | 20 KiB ping-pong p50 µs | 40 KiB exchange p50 / p99 µs '
          '| 5 MiB one way MB/s (each way) | chain token p50 / p99 µs | checked bytes |')
    print('|---|---|---|---|---|---|---|---|---|')
    for tag in ('tcp', 'stream'):
        for h in LINKS:
            rs = results(d, tag, h)
            if not rs:
                continue
            by = {}
            for x in rs:
                if 'error' in x:
                    by.setdefault('error', x['error'])
                    continue
                t, r = x['test'], x['result']
                by[(t['mode'], t['size'])] = r
            def lat(mode, size, keys=('p50', 'p99')):
                r = by.get((mode, size))
                if not r or not r.get('lat_us'):
                    return '-'
                return ' / '.join(f"{r['lat_us'][k]:.1f}" for k in keys)
            bulk = next((r for (m, s), r in by.items() if m == 'bulk'), None) if isinstance(next(iter(by), None), tuple) else None
            bulk = next((r for k, r in by.items() if isinstance(k, tuple) and k[0] == 'bulk'), None)
            mb = f"{bulk['mb_s']['client_to_server']:.0f} / {bulk['mb_s']['server_to_client']:.0f}" if bulk else '-'
            chain = next((r for k, r in by.items() if isinstance(k, tuple) and k[0] == 'chain'), None)
            ch = f"{chain['lat_us']['p50']:.0f} / {chain['lat_us']['p99']:.0f}" if chain else '-'
            checked = sum(r.get('checked_bytes', 0) for k, r in by.items() if isinstance(k, tuple))
            p = paths(next((r for k, r in by.items() if isinstance(k, tuple)), {}))
            err = f" ERROR {by['error'][:80]}" if 'error' in by else ''
            print(f"| {tag} | {h} -> {NEXT[h]} | {p}{err} | {lat('pingpong', 64)} | {lat('pingpong', 20480, ('p50',))} | "
                  f"{lat('exchange', 40960)} | {mb} | {ch} | {checked / 1e9:.2f} GB |")


def soak(d, tag='soak'):
    print('| link | rounds | checked GB | chain p50 / p99 / max µs | 40 KiB exchange p50 / p99 µs | 5 MiB MB/s | '
          'path changes | slow (>20 ms) | rewinds | crc errors | fallbacks | error |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for f in sorted(glob.glob(f'{d}/{tag}-*.jsonl')):
        h = f.rsplit('-', 1)[1][:-6]
        for x in results(d, tag, h):
            if 'error' in x:
                print(f"| {h} | | | | | | | | | | | {x['error'][:100]} |")
                continue
            r = x['result']
            st = [l['stats'] for l in r['lanes']]
            srv = [l['stats'] for l in r.get('server', {}).get('lanes', [])]
            tot = lambda k: sum(s[k] for s in st + srv)
            print(f"| {h} -> {NEXT.get(h, '?')} | {r['rounds']} | {r['checked_bytes'] / 1e9:.1f} | "
                  f"{r['chain_us']['p50']:.0f} / {r['chain_us']['p99']:.0f} / {r['chain_us']['max']:.0f} | "
                  f"{r['exchange_40k_us']['p50']:.0f} / {r['exchange_40k_us']['p99']:.0f} | "
                  f"{r['bulk_5m_mb_s']['client_to_server']:.0f} / {r['bulk_5m_mb_s']['server_to_client']:.0f} | "
                  f"{len(r.get('events', []))} | {len(r.get('slow', []))} | {tot('rewinds')} | {tot('crc_errors')} | "
                  f"{tot('fallbacks')} | |")


def faults(d):
    for f in sorted(glob.glob(f'{d}/f-*.jsonl')):
        name = os.path.basename(f)[:-6]
        for x in lines(f):
            if 'report' in x:
                continue
            if 'error' in x:
                ev = [(e.get('path') or e.get('slow'), e['t']) for e in x.get('events', [])]
                print(f"{name}: ERROR after {x['elapsed_s']:.3f} s: {x['error'][:160]}; events {ev}")
                continue
            r = x['result']
            st = [l['stats'] for l in r['lanes']]
            srv = [l['stats'] for l in r.get('server', {}).get('lanes', [])]
            keys = ('rewinds', 'naks_sent', 'naks_got', 'crc_errors', 'gaps', 'fallbacks', 'ups', 'epochs', 'stale')
            tot = {k: sum(s[k] for s in st + srv) for k in keys}
            ev = [f"{e.get('path') or ('slow ' + e.get('slow', ''))}@{e['t']:.3f}" + (f" {e['ms']:.0f}ms" if 'ms' in e else '')
                  for e in r.get('events', []) + r.get('slow', [])]
            lat = r.get('lat_us') or {}
            print(f"{name}: path {paths(r)}, checked {r['checked_bytes'] / 1e6:.0f} MB, "
                  f"p50 {lat.get('p50', 0):.0f} max {lat.get('max', 0):.0f} µs, {tot}, events {ev}")


if __name__ == '__main__':
    d, what = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'all')
    # A window that recorded its ring (ring6.py: MAC NEXT ADDR per line) names its links from it.
    if os.path.exists(f'{d}/ring.txt'):
        ring = [l.split() for l in open(f'{d}/ring.txt') if l.strip()]
        LINKS[:] = [r[0] for r in ring]
        NEXT.clear()
        NEXT.update({r[0]: r[1] for r in ring})
    if what in ('suites', 'all'):
        suites(d)
    if what in ('soak', 'all'):
        print()
        soak(d)
    if what in ('faults', 'all'):
        print()
        faults(d)
