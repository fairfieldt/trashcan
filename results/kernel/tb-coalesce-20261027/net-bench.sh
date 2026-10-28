#!/bin/bash
# thunderbolt_net on the inflight-alight cable: sockperf 64 B ping-pong round trip (10 s) and iperf3 one way each
# direction (10 s, 3 reps), as results/kernel/tb-loopback-20261006/bench.sh. Run on the Mac (this repo's machine).
# Both ends hold a 0 us CPU latency request for the run (scripts/latency-run.py in ~/trashcan-exp/kernel74/).
# usage: net-bench.sh TAG     (writes $(dirname $0)/TAG/)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); out=$here/$1; mkdir -p "$out"
A=inflight B=alight
ia=$(ssh $A "ip -4 -o addr show | awk '/thunderbolt/ && /10.99.86/ {sub(\"/.*\",\"\",\$4); print \$4}'")
ib=$(ssh $B "ip -4 -o addr show | awk '/thunderbolt/ && /10.99.86/ {sub(\"/.*\",\"\",\$4); print \$4}'")
for h in $A $B; do
    ssh $h "uname -r; i=\$(ip -o addr show | awk '/10.99.86/ {print \$2}'); echo \$i; cat /sys/module/thunderbolt_net/parameters/throttling 2>/dev/null || true; /usr/sbin/ethtool -c \$i 2>&1 | grep -E 'usecs:' | head -2 || true" > "$out/state-$h.txt" 2>&1
done
# both ends hold a 0 us CPU latency request (scripts/latency-run.py) for the whole run
for h in $A $B; do ssh $h "nohup sudo -n python3 ~/trashcan-exp/kernel74/latency-run.py sleep 150 >/dev/null 2>&1 & echo \$! > ~/trashcan-exp/kernel74/lat.pid"; done
trap 'for h in $A $B; do ssh $h "sudo -n pkill -P \$(cat ~/trashcan-exp/kernel74/lat.pid) 2>/dev/null; sudo -n kill \$(cat ~/trashcan-exp/kernel74/lat.pid) 2>/dev/null; rm -f ~/trashcan-exp/kernel74/lat.pid"; done' EXIT
ssh $B "pkill -x sockperf; pkill -x iperf3; sockperf server -i $ib --daemonize >/dev/null 2>&1"; sleep 0.5
ssh $A "sockperf ping-pong -i $ib -t 10 -m 64 --full-rtt" > "$out/sockperf-64B.txt" 2>&1
ssh $B "pkill -x sockperf"
for r in 1 2 3; do
    ssh $B "iperf3 -s -1 -D -B $ib >/dev/null"; sleep 0.5
    ssh $A "iperf3 -c $ib -J -t 10" > "$out/iperf-ab-$r.json" 2>&1
    ssh $A "iperf3 -s -1 -D -B $ia >/dev/null"; sleep 0.5
    ssh $B "iperf3 -c $ia -J -t 10" > "$out/iperf-ba-$r.json" 2>&1
done
python3 - "$out" <<'PY'
import json, re, sys, glob, os
o = sys.argv[1]
t = open(f'{o}/sockperf-64B.txt').read()
p50 = re.search(r'percentile 50\.000 =\s+([\d.]+)', t); avg = re.search(r'avg-rtt=([\d.]+)', t)
g = lambda d: [round(json.load(open(f))['end']['sum_received']['bits_per_second'] / 1e9, 2) for f in sorted(glob.glob(f'{o}/iperf-{d}-*.json'))]
s = {'rtt_p50_us': float(p50.group(1)) if p50 else None, 'rtt_avg_us': float(avg.group(1)) if avg else None,
     'gbps_inflight_to_alight': g('ab'), 'gbps_alight_to_inflight': g('ba')}
json.dump(s, open(f'{o}/summary.json', 'w')); print(os.path.basename(o), s)
PY
