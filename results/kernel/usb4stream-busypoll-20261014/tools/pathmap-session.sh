#!/bin/bash
# Six Macs under chain6.sh, CPU only, read only apart from the probes; no reloads, resets or reboots.
# (a) Each Mac's Thunderbolt path tables and adapter registers (debugfs, through stream-pathdump.sh): with the stream
#     devices closed, with every stream device held open (stream-holdopen.py, so their paths are up), and closed again
#     at the end.
# (b) The lone-frame hold on each of the six ring cables, both ways: the pos and pp probes (stream-tbcheck.py) over
#     OPENS fresh opens per direction, the client Mac taken from ORDER in turn (default abba; the first map used abab).
#     Each run's exit codes and error output are kept per end, so an open lost to ENXIO (a stock end that read a stale
#     CLOSE) is counted apart from the holds.
# pathmap.py tabulates both: per Mac and controller the stream and thunderbolt_net path ends (ring, credits, flow
# control, drop), and per direction the hold next to the path settings it went through.
# DRY=1 prints the steps.
# usage: chain6.sh stream-pathmap 15 OWNER -- bash results/kernel/usb4stream-busypoll-20261014/tools/pathmap-session.sh
set -uo pipefail
OPENS=${OPENS:-6} HOLD=${HOLD:-15} ORDER=${ORDER:-abba}
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/pathmap-$(date +%Y%m%d-%H%M%S)}
W=trashcan-exp/flashnext
MACS="incite inflight alight indite invite enlight"
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/err"
dry() { [ "${DRY:-0}" = 1 ]; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
on() { local h=$1; shift; if dry; then echo "    ssh $h: $*" >> "$OUT/steps.log"; return 0; fi; ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 "$h" "$@"; }

cleanup() {
    for h in $MACS; do
        ssh -o ConnectTimeout=10 "$h" "pkill -f '[s]tream-tbcheck.py'; pkill -f '[s]tream-holdopen.py'; true" >> "$OUT/cleanup.txt" 2>&1
    done
}
if ! dry; then
    [ -n "${TICKETS:-}" ] || { echo "run under chain6.sh (TICKETS)" >&2; exit 1; }
    ( while :; do for t in ${TICKETS//,/ }; do gpuq --on "${t%%=*}" status "${t#*=}" > /dev/null 2>&1; done; sleep 120; done ) > /dev/null 2>&1 &
    hb=$!
    trap 'kill $hb 2>/dev/null; cleanup' EXIT
    trap 'exit 130' INT TERM
fi

log "output $OUT; (a) path tables closed, open and closed again; (b) pos and pp, $OPENS opens per direction on each cable, clients in the order $ORDER"
for h in $MACS; do
    on "$h" "uname -r; cd ~/$W && md5sum stream-tbcheck.py stream-holdopen.py stream-pathdump.sh | cut -c1-12 | tr '\n' ' '; echo; /usr/local/sbin/tblink links" > "$OUT/links-$h.txt" 2>&1
done

# The ring cables, each once: A is the Mac that comes first in MACS.
cables=()
if dry; then
    cables=("incite /dev/tbstream0 inflight /dev/tbstream0" "incite /dev/tbstream1 enlight /dev/tbstream1")
else
    [ "$(for h in $MACS; do sed -n 2p "$OUT/links-$h.txt"; done | sort -u | wc -l)" -eq 1 ] || { log "STOP: the probe tools are missing or differ between the Macs"; exit 1; }
    for a in $MACS; do
        later=0
        for b in $MACS; do
            [ "$b" = "$a" ] && { later=1; continue; }
            [ $later = 1 ] || continue
            da=$(awk -F'\t' -v p="$b" 'NR > 3 && $2 == p {print ($6 == "up" && $5 == 2) ? $10 : "down"}' "$OUT/links-$a.txt")
            [ -n "$da" ] || continue
            db=$(awk -F'\t' -v p="$a" 'NR > 3 && $2 == p {print ($6 == "up" && $5 == 2) ? $10 : "down"}' "$OUT/links-$b.txt")
            case "$da $db" in
                /dev/tbstream*" "/dev/tbstream*) cables+=("$a $da $b $db") ;;
                *) log "STOP: the $a-$b link is not up on two lanes with a stream at both ends ($da, ${db:-none})"; exit 1 ;;
            esac
        done
    done
    [ ${#cables[@]} -eq 6 ] || { log "STOP: ${#cables[@]} ring cables up with streams, not 6"; exit 1; }
    for h in $MACS; do log "  $h $(head -1 "$OUT/links-$h.txt")"; done
    log "  tools $(sed -n 2p "$OUT/links-incite.txt")"
fi

snapshot() { # NAME: every Mac's path tables at once
    local pids=() h p
    for h in $MACS; do
        on "$h" "sudo -n bash ~/$W/stream-pathdump.sh" > "$OUT/paths-$1-$h.txt" 2> "$OUT/err/paths-$1-$h.txt" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    dry || log "  $1: $(for h in $MACS; do printf '%s %s lines; ' "$h" "$(grep -vc '^#' "$OUT/paths-$1-$h.txt")"; done)"
}

log "(a) path tables with the stream devices closed"
snapshot closed
log "(a) path tables with every stream device held open"
holds=()
i=0
for h in $MACS; do
    devs=$(awk -F'\t' 'NR > 3 && $6 == "up" && $10 ~ /^\/dev\/tbstream/ {printf "%s ", $10}' "$OUT/links-$h.txt")
    dry && devs="/dev/tbstream0 /dev/tbstream1"
    # The Macs close a third of a second apart, so the two ends of a cable never close together (no stale CLOSE).
    secs=$(awk -v s="$HOLD" -v i=$i 'BEGIN {printf "%.1f", s + i * 0.3}')
    on "$h" "timeout -k 5 $((HOLD + 10)) python3 ~/$W/stream-holdopen.py $secs $devs" > "$OUT/holdopen-$h.jsonl" 2> "$OUT/err/holdopen-$h.txt" &
    holds+=($!)
    i=$((i + 1))
done
sleep 4
snapshot open
for p in "${holds[@]}"; do wait "$p"; done
dry || log "  held open: $(python3 "$here/pathmap.py" --held "$OUT" 2>&1)"

run=0
probe() { # KIND CLIENT CLIENT_DEV SERVER SERVER_DEV CABLE: one fresh open at both ends; the client writes the lone frames
    local kind=$1 c=$2 cd=$3 s=$4 sd=$5 cable=$6 args ps rc rs
    run=$((run + 1))
    if [ "$kind" = pos ]; then args="8 $run"; else args="10 112 0 $run"; fi
    echo "== run $run server" >> "$OUT/err/$kind-$cable-$s.txt"
    echo "== run $run client" >> "$OUT/err/$kind-$cable-$c.txt"
    on "$s" "timeout -k 5 30 python3 ~/$W/stream-tbcheck.py $kind $sd server $args" >> "$OUT/$kind-$cable.jsonl" 2>> "$OUT/err/$kind-$cable-$s.txt" &
    ps=$!
    sleep 0.5
    on "$c" "timeout -k 5 30 python3 ~/$W/stream-tbcheck.py $kind $cd client $args" >> "$OUT/$kind-$cable.jsonl" 2>> "$OUT/err/$kind-$cable-$c.txt"
    rc=$?
    # A client that failed (an ENXIO at a stock end) leaves the server waiting for it: end the server now. A server
    # that never heard the client (its device read a stale CLOSE) is ended 3 s after the client is done.
    if [ $rc = 0 ]; then
        for _ in 1 2 3 4 5 6; do kill -0 $ps 2> /dev/null || break; sleep 0.5; done
    fi
    if kill -0 $ps 2> /dev/null; then
        on "$s" "pkill -f '[s]tream-tbcheck.py $kind'; true" > /dev/null 2>&1
    fi
    wait $ps
    rs=$?
    echo "$run $cable $kind $(date +%T) client $c $rc server $s $rs" >> "$OUT/runs.txt"
}

log "(b) the hold on each cable"
for cable in "${cables[@]}"; do
    read -r A DA B DB <<< "$cable"
    name=$A-$B
    log "$name: $A $DA <-> $B $DB"
    for kind in pos pp; do
        # ORDER: which Mac is the client in each run, repeated (abba by default, so that each Mac is client right
        # after a run with either Mac as client).
        for j in $(seq 0 $((2 * OPENS - 1))); do
            o=${ORDER:0:4}
            if [ "${o:$((j % 4)):1}" = a ]; then
                probe $kind "$A" "$DA" "$B" "$DB" "$name"
            else
                probe $kind "$B" "$DB" "$A" "$DA" "$name"
            fi
        done
    done
    dry || log "  $(python3 "$here/pathmap.py" --cable "$OUT" "$name" 2>&1 | tr '\n' ' ')"
done

log "(a) path tables with the stream devices closed again"
snapshot after
if ! dry; then
    python3 "$here/pathmap.py" "$OUT" > "$OUT/map.md" 2> "$OUT/err/pathmap.txt"
    log "map: $OUT/map.md ($(wc -l < "$OUT/map.md") lines; errors: $(wc -l < "$OUT/err/pathmap.txt") lines)"
fi
log "done"
echo "stream-pathmap done: $OUT"
