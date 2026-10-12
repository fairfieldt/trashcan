import re, collections, sys
# Sum the temporary per-dispatch timestamps (DISP lines) of the prefill chunks at position 0.
# usage: agg.py ERR BATCH [label regex]
f, b = sys.argv[1], int(sys.argv[2])
flt = sys.argv[3] if len(sys.argv) > 3 else ''
per = collections.defaultdict(list)
for l in open(f):
    if not l.startswith('DISP'):
        continue
    m = re.match(r'DISP (\S+) n=(\d+) pos=(\d+) (.*) x(\d+) ([\d.]+)$', l.strip())
    g, n, pos, lab, c, ms = m.groups()
    if int(n) != b or int(pos) != 0:
        continue
    per[(g, lab)].append((int(c), float(ms)))
tot = collections.defaultdict(float)
cnt = {}
for (g, lab), v in per.items():
    v = v[1:4] if len(v) >= 4 else v
    tot[lab] += sum(x[1] for x in v) / len(v)
    cnt[lab] = cnt.get(lab, 0) + v[0][0]
for lab, ms in sorted(tot.items(), key=lambda x: -x[1]):
    if flt and not re.search(flt, lab):
        continue
    print(f'  {lab:45s} x{cnt[lab]:3d} {ms:8.2f}  per {ms / cnt[lab]:.3f}')
