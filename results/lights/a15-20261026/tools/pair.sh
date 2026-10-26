#!/bin/bash
# A15: portlights-node running vs stopped on all six Macs, six-Mac Flash-Next decode bench (same binary and settings
# as portlights-20261026/demo-chain.sh pp2k), order on, off, off, on. Under chain6.sh (TICKETS set). First installs the
# shipped lights/ (trashcan main candidate) on all six with lights/install.sh --conductor 7465.
set -u
cd /Users/fairfieldt/dev/claude-work/trashcan-release-20261026/a15/lab/results/qwen4/chain-20261012 || exit 1
W=/home/fairfieldt/trashcan-exp/flashnext
eval "$(python3 tools/ring.py)" || exit 1
C="--gpus 0,1 --nodes $ADDRS --head-split 12 --context 32768 --ngram disk --batch 128"
OUT=/Users/fairfieldt/dev/claude-work/trashcan-release-20261026/a15/results/pair
MACS="incite indite inflight alight invite enlight"
for h in $MACS; do echo "== $h"; ssh $h 'cd ~/trashcan-exp/a15-lights && sudo -n lights/install.sh --conductor 7465'; done > $OUT/deploy.log 2>&1
nodes() { for h in $MACS; do ssh $h "sudo -n systemctl $1 portlights-node" & done; wait; }
cpu() { for h in $MACS; do echo "$h $(ssh $h systemctl show -p CPUUsageNSec --value portlights-node)"; done; }
mon() {
    for h in $MACS; do
        ssh $h 'P=/sys/devices/platform/applesmc.768; H=$(ls -d $P/hwmon/hwmon*); end=$((SECONDS+90))
            echo "# time fan1_input fan1_target pwm1_enable port_lights fancurve"
            while [ $SECONDS -lt $end ]; do
                echo "$(date +%T) $(cat $H/fan1_input) $(cat $H/fan1_target) $(cat $H/pwm1_enable) $(cat $P/port_lights) $(systemctl is-active fancurve)"
                sleep 2
            done' > $OUT/mon-$1-$h.log 2>&1 &
    done
}
for run in on1 off1 off2 on2; do
    case $run in on*) nodes start ;; off*) nodes stop ;; esac
    sleep 3
    t0=$(date +%s)
    cpu > $OUT/cpu-$run-before.txt
    mon $run
    echo "$(date +%T) $run start"
    python3 tools/q4run.py --tickets "$TICKETS" --binary trashcan-4f36bf0 --name "a15-pair-$run" --driver-host incite \
        --nodes "$NODES" --driver "bench $C --pp 2048 --tg 512 --tokens-file $W/long.ids" > "$OUT/$run.log" 2>&1
    rc=$?
    t1=$(date +%s)
    cpu > $OUT/cpu-$run-after.txt
    for h in $MACS; do echo "== $h"; ssh $h "sudo -n journalctl -u portlights-node --since @$t0 -o short-precise --no-pager"; done > $OUT/journal-$run.log 2>&1
    sleep 2
    jobs -p | xargs -r kill 2>/dev/null
    echo "$t0 $t1" > $OUT/time-$run.txt
    echo "$(date +%T) $run rc $rc: $(python3 -c 'import json,sys; d=json.loads(open(sys.argv[1]).read().strip().splitlines()[-1]); print(f"pp {d[\"pp_tok_s\"]:.1f} tg {d[\"tg_tok_s\"]:.3f} tok/s")' "$OUT/$run.log" 2>&1)"
done
nodes start
echo "a15 pair: $(for r in on1 off1 off2 on2; do python3 -c 'import json,sys; d=json.loads(open(sys.argv[1]).read().strip().splitlines()[-1]); print(f"{sys.argv[2]} {d[\"tg_tok_s\"]:.2f}", end=" ")' "$OUT/$r.log" $r 2>/dev/null; done)"
