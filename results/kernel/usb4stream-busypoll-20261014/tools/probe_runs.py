#!/usr/bin/env python3
"""Every run of a probe-session.sh or knobs-session.sh directory: per probe file, each run's client and server lines
(pp: rounds and fillers per round trip; pos: the fillers each lone frame needed), and the errors files' last lines.

usage: probe_runs.py DIR
"""
import glob, json, os, sys


def main():
    d = sys.argv[1]
    for f in sorted(glob.glob(os.path.join(d, 'pp-*.jsonl')) + glob.glob(os.path.join(d, 'pos-*.jsonl'))):
        print(os.path.basename(f))
        for line in open(f):
            if not line.startswith('{'):
                continue
            v = json.loads(line)
            if v.get('step') == 'pp':
                what = f"rounds {v.get('rounds')} fillers p50 {v.get('fillers', {}).get('p50')} frames seen {v.get('frames_seen')}"
            elif v.get('side') == 'client':
                what = f"rounds {v.get('rounds')} fillers per round {[r[2] for r in v.get('raw', [])]}"
            else:
                what = f"answered {v.get('answered')}"
            print(f"  run {v.get('run')} {v.get('host')} {v.get('side')}: {what}")
        err = f[:-len('.jsonl')] + '.err'
        if os.path.exists(err) and os.path.getsize(err):
            lines = open(err).read().splitlines()
            print(f'  errors: {len(lines)} lines, last: {lines[-1][:160]}')


if __name__ == '__main__':
    main()
