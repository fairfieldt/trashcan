#!/bin/bash
# The raw ping-pong probe (tbcheck.py pp, CPU only) on all six ring links with the streams' busy_poll off, then on, with
# no follow-up frames: does a busy-polled ring read a lone frame at once (incite-enlight held 3 frames into enlight on
# 7.2.9)? Under chain6.sh, on 7.3 with tblink's streams in place; busy_poll is put back as found. No module reloads or
# resets. DRY=1 prints the steps.
# usage: chain6.sh stream-bp1 8 OWNER -- results/kernel/usb4stream-busypoll-20261014/tools/probe-window.sh
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
tools=$repo/results/kernel/usb4stream-reliable-20261013/tools
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/bp1-$(date +%Y%m%d-%H%M%S)}
F=trashcan-exp/flashnext
MACS="incite indite inflight alight invite enlight"
CFG=/sys/kernel/config/thunderbolt/stream
mkdir -p "$OUT"
dry() { [ "${DRY:-0}" = 1 ]; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
on() { local h=$1; shift; if dry; then echo "    ssh $h: $*" >> "$OUT/steps.log"; return 0; fi; ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 "$h" "$@"; }

python3 "$tools/ring6.py" > "$OUT/ring.txt" || { log "the six links do not make a ring: $(cat "$OUT/ring.txt")"; exit 1; }
R=($(awk '{print $1}' "$OUT/ring.txt"))
next_of() { awk -v h="$1" '$1 == h {print $2}' "$OUT/ring.txt"; }
busy() { # 0|1: every stream end's busy_poll (devices closed), read back
    local h
    for h in $MACS; do
        on "$h" "for d in $CFG/*/trashcan; do echo $1 | sudo -n tee \$d/busy_poll > /dev/null; done; grep . $CFG/*/trashcan/busy_poll" >> "$OUT/busy-$1.txt" 2>&1
    done
    dry || log "  busy_poll: $(awk -F: '{print $NF}' "$OUT/busy-$1.txt" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
}
if ! dry; then
    [ -n "${TICKETS:-}" ] || { echo "run under chain6.sh (TICKETS)" >&2; exit 1; }
    ( while :; do for t in ${TICKETS//,/ }; do gpuq --on "${t%%=*}" status "${t#*=}" > /dev/null 2>&1; done; sleep 120; done ) > /dev/null 2>&1 &
    trap 'for j in $(jobs -p); do pkill -P $j 2>/dev/null; kill $j 2>/dev/null; done
        for h in $MACS; do ssh "$h" "pkill -f \"[s]tream-tbcheck.py\"; for d in $CFG/*/trashcan; do echo 0 | sudo -n tee \$d/busy_poll > /dev/null; done" 2>/dev/null; done' EXIT
    trap 'exit 130' INT TERM
fi

log "output $OUT"
log "ring: $(awk '{print $1 "->" $3}' "$OUT/ring.txt" | tr '\n' ' ')"
for h in $MACS; do
    on "$h" "uname -r; test -f ~/$F/stream-tbcheck.py && /usr/local/sbin/tblink links" > "$OUT/links-$h.txt" 2>&1 || { log "$h: tbcheck or tblink missing"; exit 1; }
    dry && continue
    awk -F'\t' 'NR > 2 && ($6 != "up" || $5 != 2 || $10 == "-") { exit 1 }' "$OUT/links-$h.txt" || { log "$h: a link is not up on two lanes with its stream: $(tail -n +3 "$OUT/links-$h.txt" | cut -f2,5,6,10 | tr '\n' ';')"; exit 1; }
done
dry || log "kernels: $(head -1 -q "$OUT"/links-*.txt | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
dev_of() { awk -F'\t' -v p="$2" 'NR > 2 && $2 == p {print $10}' "$OUT/links-$1.txt"; }

probe() { # tag follow rounds: single 112-byte frames there and back on every link at once
    local tag=$1 follow=$2 n=$3 h nx pids=""
    for h in "${R[@]}"; do
        nx=$(next_of "$h")
        on "$nx" "timeout -k 5 60 python3 ~/$F/stream-tbcheck.py pp $(dev_of "$nx" "$h") server $n 112 $follow 41" >> "$OUT/probe-$tag.jsonl" 2>> "$OUT/probe-$tag.err" &
        pids="$pids $!"
    done
    sleep 0.5
    for h in "${R[@]}"; do
        nx=$(next_of "$h")
        on "$h" "timeout -k 5 60 python3 ~/$F/stream-tbcheck.py pp $(dev_of "$h" "$nx") client $n 112 $follow 41" >> "$OUT/probe-$tag.jsonl" 2>> "$OUT/probe-$tag.err" &
        pids="$pids $!"
    done
    wait $pids
    dry || python3 "$tools/probe_table.py" "$OUT/probe-$tag.jsonl" | tee -a "$OUT/steps.log"
}

log "1. busy_poll 0 (as created), no follow-up frames"
busy 0
probe bp0 0 25
log "2. busy_poll 1, no follow-up frames"
busy 1
probe bp1 0 200
log "3. busy_poll back to 0"
busy 0
log "done"
echo "stream-bp1 done: $OUT"
