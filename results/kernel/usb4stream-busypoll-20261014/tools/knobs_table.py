#!/usr/bin/env python3
"""Tables of a knobs-session.sh run: per switch setting, each probe's hold over every fresh open (pp: fillers per round
trip with lone frames both ways; pos: the fillers a lone frame needed into the other end), by client, and the churn
sessions (passed, failed epochs, one-way epochs).

usage: knobs_table.py OUTDIR
"""
import collections, json, os, re, sys


def holds(path, probe):
    by = collections.OrderedDict()
    for line in open(path) if os.path.exists(path) else []:
        if not line.startswith('{'):
            continue
        v = json.loads(line)
        if v.get('side') != 'client':
            continue
        if probe == 'pos':
            h = sorted(set(r[2] for r in v.get('raw', [])))
            by.setdefault(v['host'], []).append('/'.join(map(str, h)) if h else 'x')
        else:
            p50 = v.get('fillers', {}).get('p50')
            by.setdefault(v['host'], []).append('x' if p50 is None else str(p50))
    return by


def main():
    d = sys.argv[1]
    steps = open(os.path.join(d, 'steps.log')).read()
    names = dict(re.findall(r'\] (s\d+)\. (.*)', steps))
    print('| setting | probe | per open, client A | per open, client B | churn passed | failed epochs | one-way |')
    print('|---|---|---|---|---|---|---|')
    for tag in sorted(names, key=lambda t: int(t[1:])):
        churn = os.path.join(d, f'churn-{tag}.jsonl')
        passed = failed = oneway = '-'
        if os.path.exists(churn):
            rows = [json.loads(l) for l in open(churn) if l.startswith('{')]
            sess = [r for r in rows if 'session' in r]
            passed = f"{sum(r['pass'] for r in sess)}/{len(sess)}"
            failed = str(sum(r['failed_epochs'] for r in sess))
            oneway = str(sum(any('reads none of our frames' in s for s in r.get('stream_lines', [])) for r in sess))
        for probe in ('pp', 'pos'):
            by = holds(os.path.join(d, f'{probe}-{tag}.jsonl'), probe)
            if not by:
                continue
            cols = [f"{h}: {' '.join(x)}" for h, x in by.items()] + ['', '']
            print(f'| {names[tag]} | {probe} | {cols[0]} | {cols[1]} | {passed} | {failed} | {oneway} |')


if __name__ == '__main__':
    main()
