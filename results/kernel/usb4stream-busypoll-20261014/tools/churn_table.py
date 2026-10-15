#!/usr/bin/env python3
"""Summary of a `link-test churn` run (its JSON lines) and the server's log of it: per session kind, sessions, passed,
time to stream (link start to stream up), failed epochs; then every session that did not pass.

usage: churn_table.py CHURN.jsonl [SERVE.log]
       churn_table.py --line CHURN.jsonl     one line: passed, time to stream, failed and one-way epochs
"""
import json, sys

# The link's reasons for giving up an epoch that carried one way only (either direction).
ONE_WAY = ('reads none of our frames', 'frames do not arrive')


def line(path):
    rows, summary = [], None
    for l in open(path):
        if l.startswith('{'):
            v = json.loads(l)
            if 'session' in v:
                rows.append(v)
            elif 'churn' in v:
                summary = v
    ups = sorted(x['up_ms'] for x in rows if x['up_ms'] is not None)
    oneway = sum(any(w in s for s in x.get('stream_lines', []) for w in ONE_WAY) for x in rows)
    f = lambda x: '-' if x is None else f'{x:.0f}'
    return (f"{'pass' if summary and summary.get('pass') else 'FAIL'}: {sum(x['pass'] for x in rows)}/{len(rows)} passed"
            f"{'' if summary else ' (no summary)'}, up p50/p99/max {f(pct(ups, .5))}/{f(pct(ups, .99))}/"
            f"{f(ups[-1] if ups else None)} ms, {sum(x['failed_epochs'] for x in rows)} failed epochs, {oneway} one-way")

KIND = {'n': 'normal exit', 'k': 'client killed', 's': 'server exits'}


def pct(v, p):
    return v[min(len(v) - 1, round((len(v) - 1) * p))] if v else None


def main():
    if sys.argv[1] == '--line':
        print(line(sys.argv[2]))
        return
    rows, summary = [], None
    for l in open(sys.argv[1]):
        v = json.loads(l)
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
