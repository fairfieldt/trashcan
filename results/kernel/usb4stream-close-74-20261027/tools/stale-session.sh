#!/bin/bash
# closestale.py on the inflight-alight cable, REPS rounds each way for each MODE (nb, block) and busy_poll (0, 1).
# A round, with X the end under test and Y the other: X holds its device open without reading; Y opens and closes
# (a CLOSE to an open end); X closes; then X opens a new session ("new": waits up to 1 s for data) while Y, 0.2 s
# later, opens and writes one frame ("peer"). Pass: X reads Y's frame and its write succeeds.
# usage: stale-session.sh OUT [REPS]
set -uo pipefail
OUT=$1 REPS=${2:-10}
mkdir -p "$OUT"
T=trashcan-exp/kernel74/closestale.py
dev() { ssh "$1" "/usr/local/sbin/tblink links | awk -F'\t' -v p=$2 '\$2 == p {print \$10}'"; }
DI=$(dev inflight alight) DA=$(dev alight inflight)
echo "inflight $DI, alight $DA" | tee "$OUT/devs.txt"
for bp in 0 1; do
    for h in inflight alight; do ssh $h "for f in /sys/kernel/config/thunderbolt/stream/*/trashcan/busy_poll; do echo $bp | sudo -n tee \$f >/dev/null; done"; done
    for mode in nb block; do
        for dir in "inflight $DI alight $DA" "alight $DA inflight $DI"; do
            set -- $dir; X=$1 DX=$2 Y=$3 DY=$4
            for r in $(seq "$REPS"); do
                tag="bp$bp $mode $X r$r"
                ssh $X "timeout 10 python3 ~/$T hold $DX 0.6" > "$OUT/.hold" 2>&1 &
                sleep 0.2
                ssh $Y "timeout 10 python3 ~/$T openclose $DY" > "$OUT/.oc" 2>&1
                wait
                ssh $X "timeout 10 python3 ~/$T new $DX $mode 1.0" > "$OUT/.new" 2>&1 &
                sleep 0.2
                ssh $Y "timeout 10 python3 ~/$T peer $DY $mode 0.1" > "$OUT/.peer" 2>&1
                wait
                python3 - "$tag" "$OUT" >> "$OUT/rounds.jsonl" <<'PY'
import json, sys
tag, o = sys.argv[1], sys.argv[2]
r = {'tag': tag}
for k in ('hold', 'oc', 'new', 'peer'):
    t = open(f'{o}/.{k}').read().strip()
    try:
        r[k] = json.loads(t.splitlines()[-1])
    except Exception:
        r[k] = {'raw': t[-300:]}
n = r['new']
r['pass'] = n.get('bytes', 0) > 0 and n.get('write') == 'ok'
print(json.dumps(r))
PY
            done
        done
    done
done
python3 - "$OUT/rounds.jsonl" <<'PY' | tee "$OUT/summary.txt"
import json, sys, collections
c = collections.defaultdict(lambda: [0, 0]); ex = {}
for l in open(sys.argv[1]):
    v = json.loads(l); k = ' '.join(v['tag'].split()[:3]); c[k][0] += v['pass']; c[k][1] += 1
    if not v['pass'] and k not in ex: ex[k] = v['new']
for k, (p, n) in c.items(): print(f'{k}: {p}/{n} pass')
for k, v in ex.items(): print('first failure', k, json.dumps(v))
PY
