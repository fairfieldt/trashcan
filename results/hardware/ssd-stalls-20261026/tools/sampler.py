#!/usr/bin/env python3
"""Sample /proc/diskstats for one disk and /proc/pressure/io every PERIOD s until killed.
Columns: t rMB wMB rIO wIO flushes flush_ms inflight io_some_total_us(delta)"""
import sys, time
dev = sys.argv[1]; period = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
def ds():
    for l in open('/proc/diskstats'):
        f = l.split()
        if f[2] == dev:
            return dict(r=int(f[3]), rs=int(f[5]), w=int(f[7]), ws=int(f[9]), inf=int(f[11]),
                        fl=int(f[18]) if len(f) > 18 else 0, flms=int(f[19]) if len(f) > 19 else 0)
def psi():
    return int(open('/proc/pressure/io').readline().split('total=')[1])
p, pp, t0 = ds(), psi(), time.time()
print('t rMBps wMBps rIOps wIOps flush_ps flush_ms_per inflight io_some_pct', flush=True)
while True:
    time.sleep(period)
    c, cp, t = ds(), psi(), time.time()
    dt = period
    nfl = c['fl'] - p['fl']
    print(f"{t:.2f} {(c['rs']-p['rs'])*512/1e6/dt:.1f} {(c['ws']-p['ws'])*512/1e6/dt:.1f} "
          f"{(c['r']-p['r'])/dt:.0f} {(c['w']-p['w'])/dt:.0f} {nfl/dt:.1f} "
          f"{((c['flms']-p['flms'])/nfl) if nfl else 0:.0f} {c['inf']} {(cp-pp)/(dt*1e4):.0f}", flush=True)
    p, pp = c, cp
