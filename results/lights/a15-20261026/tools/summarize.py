import json, sys, os, glob
d = sys.argv[1]
macs = "incite indite inflight alight invite enlight".split()
for r in ["on1", "off1", "off2", "on2"]:
    j = json.loads(open(f"{d}/{r}.log").read().strip().splitlines()[-1])
    t0, t1 = map(int, open(f"{d}/time-{r}.txt").read().split())
    b = dict(l.split() for l in open(f"{d}/cpu-{r}-before.txt"))
    a = dict(l.split() for l in open(f"{d}/cpu-{r}-after.txt"))
    cpu = []
    for m in macs:
        try: cpu.append((int(a[m]) - int(b[m])) / 1e9)
        except: cpu.append(None)
    clk = [l for l in open(f"{d}/{r}.log") if l.startswith("[clocks]")]
    modes = open(f"{d}/journal-{r}.log").read().count("blinkenlights")
    fans = []
    for m in macs:
        rows = [l.split() for l in open(f"{d}/mon-{r}-{m}.log") if l[0].isdigit()]
        fans.append((min(int(x[2]) for x in rows), set(x[3] for x in rows), set(x[5] for x in rows), set(x[4] for x in rows)))
    print(f"{r}: pp {j['pp_tok_s']:.1f} tok/s, tg {j['tg_tok_s']:.3f} tok/s, {t1-t0} s; node CPU s per Mac {['-' if c is None else round(c,2) for c in cpu]}; 'blinkenlights' journal lines {modes}")
    print("   fan target min / pwm1_enable / fancurve / port_lights:", [(f[0], ','.join(f[1]), ','.join(f[2]), ','.join(f[3])) for f in fans])
    for c in clk: print("  ", c.strip()[:200])
