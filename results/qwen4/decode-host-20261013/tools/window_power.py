#!/usr/bin/env python3
"""Mean SMC PSTR watts per Mac over the windows a session log marks (`window NAME start HH:MM:SS` ... `window NAME end
HH:MM:SS`, this Mac's local time), from the netconsole telemetry (CRASHCAPTURE lines).
usage: window_power.py SESSION_LOG NETCONSOLE_LOG [DATE]   (DATE YYYY-MM-DD of the session, default today)"""
import datetime, json, re, statistics, sys
from collections import defaultdict

log, net = sys.argv[1:3]
day = sys.argv[3] if len(sys.argv) > 3 else datetime.date.today().isoformat()
marks = defaultdict(dict)
for line in open(log):
    m = re.match(r'window (\S+) (start|end) (\d\d:\d\d:\d\d)', line)
    if m:
        marks[m.group(1)][m.group(2)] = f'{day}T{m.group(3)}'
lines = [l for l in open(net, errors='replace') if 'CRASHCAPTURE' in l and 'telemetry' in l]
for name, w in marks.items():
    if 'start' not in w or 'end' not in w:
        continue
    watts = defaultdict(list)
    for l in lines:
        if w['start'] <= l[:19] <= w['end']:
            d = json.loads(l.split('CRASHCAPTURE ', 1)[1])
            v = ((d.get('smc') or {}).get('PSTR') or {}).get('watts')
            if v is not None:
                watts[d['host']].append(v)
    per = {h: statistics.mean(v) for h, v in sorted(watts.items())}
    print(f"{name:20s} {w['start'][11:]}-{w['end'][11:]}  " + ' '.join(f'{h} {v:.0f}' for h, v in per.items())
          + f"  six Macs {sum(per.values()):.0f} W")
