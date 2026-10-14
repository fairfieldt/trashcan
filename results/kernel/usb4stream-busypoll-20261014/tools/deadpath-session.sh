#!/bin/bash
# One cable, A <-> B, under leases on both Macs (pair2.sh), CPU only; no reloads, resets or reboots: does writing toward
# an end whose device is closed leave that direction dead once the end opens? Each round, the early end opens and writes
# K data frames while the late end is still closed; the late end opens OPEN s later; then both write 20 test frames and
# count what arrives (deadpath.py). K 0 (control), 4, 16, 64, 300 with OPEN 0.1 s, and 16, 300 with OPEN 1 s; each
# twice, with A early, then B early. SPECS="K OPEN|K OPEN..." and REPS=N choose other rounds (K -1: a frame every
# millisecond until the test, so frames arrive while the late end opens). DRY=1 prints the steps.
# usage: pair2.sh stream-deadpath 10 OWNER A B -- bash results/kernel/usb4stream-busypoll-20261014/tools/deadpath-session.sh A B
set -uo pipefail
A=$1 B=$2
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/deadpath-$(date +%Y%m%d-%H%M%S)}
W=trashcan-exp/flashnext
DP=$W/stream-deadpath.py
DEV_A=/dev/tbstreamA DEV_B=/dev/tbstreamB    # from the live links
mkdir -p "$OUT"
dry() { [ "${DRY:-0}" = 1 ]; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
on() { local h=$1; shift; if dry; then echo "    ssh $h: $*" >> "$OUT/steps.log"; return 0; fi; ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 "$h" "$@"; }

cleanup() {
    for h in $A $B; do ssh -o ConnectTimeout=10 "$h" "pkill -f '[s]tream-deadpath.py'; true" >> "$OUT/cleanup.txt" 2>&1; done
}
if ! dry; then
    [ -n "${TICKETS:-}" ] || { echo "run under pair2.sh (TICKETS)" >&2; exit 1; }
    ( while :; do for t in ${TICKETS//,/ }; do gpuq --on "${t%%=*}" status "${t#*=}" > /dev/null 2>&1; done; sleep 120; done ) > /dev/null 2>&1 &
    hb=$!
    trap 'kill $hb 2>/dev/null; cleanup' EXIT
    trap 'exit 130' INT TERM
fi

log "output $OUT"
for h in $A $B; do
    on "$h" "uname -r; date +%s.%N; /usr/local/sbin/tblink links" > "$OUT/links-$h.txt" 2>&1
done
if ! dry; then
    DEV_A=$(awk -F'\t' -v p="$B" 'NR > 3 && $2 == p && $6 == "up" && $5 == 2 {print $10}' "$OUT/links-$A.txt")
    DEV_B=$(awk -F'\t' -v p="$A" 'NR > 3 && $2 == p && $6 == "up" && $5 == 2 {print $10}' "$OUT/links-$B.txt")
    log "  $A $(head -1 "$OUT/links-$A.txt") $DEV_A <-> $B $(head -1 "$OUT/links-$B.txt") $DEV_B"
    case "$DEV_A$DEV_B" in /dev/tbstream*/dev/tbstream*) ;; *) log "STOP: the $A-$B link is not up on two lanes with a stream at both ends"; exit 1 ;; esac
fi
run=0
for early in $A $B; do
    if [ "$early" = "$A" ]; then late=$B de=$DEV_A dl=$DEV_B; else late=$A de=$DEV_B dl=$DEV_A; fi
    log "$early early, $late late"
    IFS='|' read -r -a specs <<< "${SPECS:-0 0.1|4 0.1|16 0.1|64 0.1|300 0.1|16 1.0|300 1.0}"
    for spec in "${specs[@]}"; do
        set -- $spec
        k=$1 open=$2
        for rep in $(seq "${REPS:-2}"); do
            run=$((run + 1))
            start=$(python3 -c 'import time; print(f"{time.time() + 1.5:.3f}")')
            test=$(python3 -c "print($open + 0.4)")
            end=$(python3 -c "print($open + 1.4)")
            on "$early" "timeout 20 python3 ~/$DP early $de $start $k $open $test $end $run" >> "$OUT/deadpath.jsonl" 2>> "$OUT/deadpath.err" &
            pe=$!
            on "$late" "timeout 20 python3 ~/$DP late $dl $start $k $open $test $end $run" >> "$OUT/deadpath.jsonl" 2>> "$OUT/deadpath.err" &
            pl=$!
            wait $pe $pl
        done
    done
done
dry || python3 - "$OUT/deadpath.jsonl" <<'EOF' | tee -a "$OUT/steps.log"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
by = {}
for r in rows:
    by.setdefault(r['run'], {})[r['role']] = r
print(f"{'run':>3} {'early':8} {'K':>4} {'open s':>6} {'burst sent':>10} {'burst read':>10} {'early->late':>11} {'late->early':>11}")
for run, d in sorted(by.items()):
    e, l = d.get('early'), d.get('late')
    if not (e and l):
        print(f'{run:3} incomplete: {d}')
        continue
    print(f"{run:3} {e['host']:8} {e['k']:4} {e['open']:6} {e.get('burst_written', 0):10} {l['burst_read']:10} "
          f"{l['test_read']:8}/20 {e['test_read']:8}/20")
EOF
log "done"
echo "stream-deadpath done: $OUT"
