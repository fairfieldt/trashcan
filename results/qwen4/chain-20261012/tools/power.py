# Per-host SMC power from netconsole telemetry between two local times. usage: power.py LOG START END (HH:MM:SS)
import json, sys, collections
log, start, end = sys.argv[1:4]
rows = collections.defaultdict(list)
for line in open(log, errors='replace'):
    if 'telemetry' not in line or 'CRASHCAPTURE' not in line: continue
    t = line[11:19]
    if not (start <= t <= end): continue
    d = json.loads(line.split('CRASHCAPTURE ', 1)[1])
    smc = d.get('smc') or {}
    w = {k: v.get('watts') for k, v in smc.items() if isinstance(v, dict) and 'watts' in v}
    rows[d['host']].append((t, w))
for h, r in sorted(rows.items()):
    keys = sorted({k for _, w in r for k in w})
    avg = {k: sum(w.get(k, 0) for _, w in r) / len(r) for k in keys}
    mx = {k: max(w.get(k, 0) for _, w in r) for k in keys}
    print(h, len(r), 'samples', ' '.join(f'{k} avg {avg[k]:.0f} max {mx[k]:.0f}' for k in keys))
