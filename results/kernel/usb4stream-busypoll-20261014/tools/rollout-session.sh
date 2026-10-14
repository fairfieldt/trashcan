#!/bin/bash
# Roll NEW (7.3.0-rc7-p2p-s2: patches 0001-0007) out to the six Mac Pros under chain6.sh (all six leases):
#  1. state: kernels, the NEW packages in ~/DEBS_DIR (sha256), kernel-boot.sh and the churn build BIN (md5), links;
#  2. install NEW on all six for the next boot only (kernel-boot.sh once), then wait for $OUT/go-reboot (the
#     coordinator is told right before), reboot all six at once and wait until each is up on NEW;
#  3. wait for $OUT/links-up (the coordinator resets Thunderbolt on all six);
#  4. the short check on each Mac under its lease (check.sh: system state, VBIOS, fans, DPM, xgpu p2p-verify, dmesg);
#  5. every ring cable up on two lanes with a stream at both ends, then link-test churn on all six links at once:
#     on each cable, SESSIONS (50) sessions from the Mac before it in the ring to the Mac after it, whose link-test
#     server runs in a loop that restarts it after each crash; a link passes when every session reached its stream
#     within 100 ms of its link's start with no failed epoch;
#  6. only if every link passes: NEW as GRUB's default on all six (kernel-boot.sh default).
# GATES=auto skips the two waits (simtest only). DRY=1 prints the steps.
# usage: chain6.sh stream-rollout 30 OWNER -- bash results/kernel/usb4stream-busypoll-20261014/tools/rollout-session.sh
set -uo pipefail
NEW=${NEW:-7.3.0-rc7-p2p-s2} DEBS_DIR=${DEBS_DIR:-kernel/7.3/debs-s2} SESSIONS=${SESSIONS:-50}
BIN=${BIN:-trashcan-exp/flashnext/trashcan-1687af5}
BN=$(basename "$BIN")
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/rollout-$(date +%Y%m%d-%H%M%S)}
K=trashcan-exp/kernel73
W=trashcan-exp/flashnext
RING="incite inflight alight indite invite enlight"
PORT0=7090
mkdir -p "$OUT"
dry() { [ "${DRY:-0}" = 1 ]; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
on() { local h=$1; shift; if dry; then echo "    ssh $h: $*" >> "$OUT/steps.log"; return 0; fi; ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 "$h" "$@"; }
ticket() { local t a=${TICKETS:-}; for t in ${a//,/ }; do [ "${t%%=*}" = "$1" ] && echo "${t#*=}"; done; }
next() { local r=($RING) i; for i in 0 1 2 3 4 5; do [ "${r[$i]}" = "$1" ] && echo "${r[$(((i + 1) % 6))]}"; done; }
port() { local r=($RING) i; for i in 0 1 2 3 4 5; do [ "${r[$i]}" = "$1" ] && echo $((PORT0 + i)); done; }

cleanup() {
    local h
    for h in $RING; do
        ssh -o ConnectTimeout=10 "$h" "touch ~/$W/stream-roll.stop; pkill -f '[l]ink-test serve --listen 0.0.0.0:709'; pkill -f '[${BN:0:1}]${BN:1} link-test'; true" >> "$OUT/cleanup.txt" 2>&1
    done
}
if ! dry; then
    [ -n "${TICKETS:-}" ] || { echo "run under chain6.sh (TICKETS)" >&2; exit 1; }
    ( while :; do for t in ${TICKETS//,/ }; do gpuq --on "${t%%=*}" status "${t#*=}" > /dev/null 2>&1; done; sleep 120; done ) > /dev/null 2>&1 &
    hb=$!
    trap 'kill $hb 2>/dev/null; cleanup' EXIT
    trap 'exit 130' INT TERM
fi

gate() { # NAME MINUTES: wait for $OUT/NAME (GATES=auto: at once)
    log "waiting for $OUT/$1 (up to $2 min)"
    if [ "${GATES:-wait}" = auto ] || dry; then touch "$OUT/$1"; fi
    local t=0
    while [ ! -f "$OUT/$1" ]; do
        [ $t -ge $(($2 * 60)) ] && { log "STOP: no $1 after $2 min"; return 1; }
        sleep 5
        t=$((t + 5))
    done
    log "  $1"
}

state() { # TAG: kernel, packages, tools and links of every Mac (state-TAG-HOST.txt); STOP unless all six ring cables are up
    local h ok=1 n b
    for h in $RING; do
        on "$h" "uname -r; (cd ~/$DEBS_DIR && sha256sum linux-image-${NEW}_*.deb linux-headers-${NEW}_*.deb | cut -c1-12 | tr '\n' ' '); echo; md5sum ~/$K/kernel-boot.sh ~/$K/check.sh ~/$BIN | cut -c1-12 | tr '\n' ' '; echo; /usr/local/sbin/tblink links" > "$OUT/state-$1-$h.txt" 2>&1
    done
    dry && return 0
    for h in $RING; do
        n=$(next "$h")
        b=$(awk -F'\t' -v p="$n" 'NR > 4 && $2 == p {print ($6 == "up" && $5 == 2 && $10 ~ /^\/dev\/tbstream/) ? "up" : $6 "/" $5 "/" $10}' "$OUT/state-$1-$h.txt")
        [ "$b" = up ] || { ok=0; log "  $h -> $n: ${b:-no link}"; }
    done
    log "  kernels: $(for h in $RING; do printf '%s %s; ' "$h" "$(head -1 "$OUT/state-$1-$h.txt")"; done)"
    log "  packages: $(for h in $RING; do sed -n 2p "$OUT/state-$1-$h.txt"; done | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    log "  kernel-boot.sh, check.sh, $(basename "$BIN"): $(for h in $RING; do sed -n 3p "$OUT/state-$1-$h.txt"; done | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    [ $ok = 1 ] || { log "STOP: not every ring cable is up on two lanes with a stream at both ends"; return 1; }
    log "  all six ring cables up on two lanes with a stream at both ends"
}

log "output $OUT; NEW $NEW from ~/$DEBS_DIR, churn build ~/$BIN, $SESSIONS sessions per link"
log "1. state on the running kernels"
state before || exit 1
if ! dry; then
    for h in $RING; do
        [ "$(sed -n 2p "$OUT/state-before-$h.txt" | wc -w)" -eq 2 ] || { log "STOP: $h has no $NEW packages in ~/$DEBS_DIR"; exit 1; }
    done
fi

log "2. install $NEW on all six for the next boot only"
pids=()
for h in $RING; do
    on "$h" "DEBS=~/$DEBS_DIR nice -n 10 ~/$K/kernel-boot.sh once $NEW && sudo -n grub-editenv list" > "$OUT/install-$h.txt" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for h in $RING; do
    dry || log "  $h: $(grep -E 'next_entry|installed' "$OUT/install-$h.txt" | sed 's/.*>gnulinux-//; s/-advanced-.*//' | tr '\n' ' ')"
done
[ $fail = 0 ] || { log "STOP: an install failed; nothing was rebooted"; exit 1; }
touch "$OUT/ready-to-reboot"
gate go-reboot 45 || exit 1
log "3. reboot all six into $NEW"
for h in $RING; do on "$h" "sudo -n systemctl reboot" > /dev/null 2>&1; done
first=${REBOOT_WAIT:-30}
dry || sleep "$first"
for h in $RING; do
    t=0
    until dry || [ "$(on "$h" 'uname -r' 2> /dev/null)" = "$NEW" ]; do
        [ $t -ge 900 ] && { log "STOP: $h not back on $NEW after 15 min"; exit 1; }
        sleep 5
        t=$((t + 5))
    done
    dry || log "  $h up on $NEW after $((t + first)) s"
done
touch "$OUT/waiting-for-links"
gate links-up 60 || exit 1

log "4. the short check on each Mac (check.sh under its lease)"
pids=()
for h in $RING; do
    on "$h" "cd ~/$K && gpuq run $(ticket "$h") -- ./check.sh" > "$OUT/check-$h.txt" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
for h in $RING; do
    dry || log "  $h: $(grep -E '^dmesg:|^exit |vbios' "$OUT/check-$h.txt" | tr '\n' ' ' | cut -c1-240)"
done

log "5. churn on all six links at once"
state after || exit 1
for h in $RING; do
    n=$(next "$h") p=$(port "$h")
    # The server's log lines from here on are this churn's (counted before it starts).
    on "$n" "cat ~/$W/stream-roll-serve-$p.log 2> /dev/null | wc -l" > "$OUT/serve-start-$p.txt" 2> /dev/null
    # Only the loop goes to the background, its output in the log, so ssh returns at once.
    on "$n" "rm -f ~/$W/stream-roll.stop; pgrep -f '^[^ ]*/$BN link-test serve --listen 0.0.0.0:$p' > /dev/null || { setsid nohup bash -c 'while [ ! -f ~/$W/stream-roll.stop ]; do ~/$BIN link-test serve --listen 0.0.0.0:$p; sleep 0.05; done' >> ~/$W/stream-roll-serve-$p.log 2>&1 < /dev/null & }"
done
pids=()
for h in $RING; do
    n=$(next "$h") p=$(port "$h")
    bus=$(awk -F'\t' -v q="$n" 'NR > 4 && $2 == q {print $3}' "$OUT/state-after-$h.txt" 2> /dev/null)
    on "$h" "timeout -k 10 900 ~/$BIN link-test churn --connect $n.tb${bus:-0}:$p --sessions $SESSIONS --label roll-$h-$n" > "$OUT/churn-$h-$n.jsonl" 2> "$OUT/churn-$h-$n.err" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
for h in $RING; do
    n=$(next "$h") p=$(port "$h")
    m=$(tr -dc 0-9 < "$OUT/serve-start-$p.txt" 2> /dev/null)
    on "$n" "touch ~/$W/stream-roll.stop; pkill -f '[l]ink-test serve --listen 0.0.0.0:$p'; sleep 0.5; tail -n +$((${m:-0} + 1)) ~/$W/stream-roll-serve-$p.log" > "$OUT/serve-$h-$n.log" 2>&1
done
allpass=1
for h in $RING; do
    n=$(next "$h")
    s=$(tail -1 "$OUT/churn-$h-$n.jsonl" 2> /dev/null)
    case "$s" in *'"pass":true'* | *'"pass": true'*) ;; *) allpass=0 ;; esac
    dry || log "  $h -> $n: $(python3 "$(dirname "$0")/churn_table.py" --line "$OUT/churn-$h-$n.jsonl" 2>&1 | tail -1); server: $(grep -c ' not opened (' "$OUT/serve-$h-$n.log") failed epochs"
done
dry && allpass=1

if [ $allpass = 1 ]; then
    log "6. every link passed: $NEW as GRUB's default on all six"
    for h in $RING; do
        on "$h" "DEBS=~/$DEBS_DIR ~/$K/kernel-boot.sh default $NEW" > "$OUT/default-$h.txt" 2>&1 || log "  $h: kernel-boot.sh default failed: $(tail -2 "$OUT/default-$h.txt" | tr '\n' ' ')"
        dry || log "  $h: $(tail -2 "$OUT/default-$h.txt" | sed 's/.*>gnulinux-//; s/-advanced-[^"]*"//' | tr '\n' ' ')"
    done
    log "done"
    echo "stream-rollout done: every link passed; GRUB default $NEW on all six"
else
    log "STOP: not every link passed; GRUB's default is unchanged ($NEW boots only once)"
    echo "stream-rollout: not every link passed; GRUB default unchanged: $OUT"
    exit 1
fi
