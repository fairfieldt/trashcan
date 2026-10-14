#!/usr/bin/env python3
"""Summary of a `link-test churn` run (its JSON lines) and the server's log of it: per session kind, sessions, passed,
time to stream (link start to stream up), failed epochs; then every session that did not pass.

usage: churn_table.py CHURN.jsonl [SERVE.log]
"""
import json, sys

KIND = {'n': 'normal exit', 'k': 'client killed', 's': 'server exits'}


def pct(v, p):
    return v[min(len(v) - 1, round((len(v) - 1) * p))] if v else None


def main():
    rows, summary = [], None
    for line in open(sys.argv[1]):
        v = json.loads(line)
        if 'session' in v:
            rows.append(v)
        elif 'churn' in v:
            summary = v
    print(f"{'kind':14} {'sessions':>8} {'passed':>6} {'up p50':>8} {'up p90':>8} {'up max':>8} {'failed epochs':>13}")
    for k in 'nks':
        r = [x for x in rows if x['kind'] == k]
        if not r:
            continue
        ups = sorted(x['up_ms'] for x in r if x['up_ms'] is not None)
        f = lambda x: '-' if x is None else f'{x:.1f}'
        print(f"{KIND[k]:14} {len(r):8} {sum(x['pass'] for x in r):6} {f(pct(ups, .5)):>8} {f(pct(ups, .9)):>8} "
              f"{f(ups[-1] if ups else None):>8} {sum(x['failed_epochs'] for x in r):13}")
    ups = sorted(x['up_ms'] for x in rows if x['up_ms'] is not None)
    print(f"{'all':14} {len(rows):8} {sum(x['pass'] for x in rows):6} {pct(ups, .5) or 0:8.1f} {pct(ups, .9) or 0:8.1f} "
          f"{(ups[-1] if ups else 0):8.1f} {sum(x['failed_epochs'] for x in rows):13}")
    if summary:
        print('summary:', json.dumps(summary))
    if len(sys.argv) > 2:
        log = open(sys.argv[2]).read().splitlines()
        print(f"server: {sum(' not opened (' in l for l in log)} failed epochs, {sum('exiting at' in l for l in log)} "
              f"exits, {sum('opening the stream device took' in l for l in log)} slow opens")
        for l in log:
            if ' not opened (' in l or 'took' in l or 'on TCP for' in l:
                print('  ' + l[:160])
    bad = [x for x in rows if not x['pass']]
    if bad:
        print('not passed:')
    for x in bad:
        lines = '; '.join(s[s.find(': ') + 2:][:80] for s in x.get('stream_lines', []))
        print(f"  {x['session']:4} {x['kind']} up {x['up_ms']} failed {x['failed_epochs']} rc {x['rc']} {x['ms']:.0f} ms | {lines}")


if __name__ == '__main__':
    main()
