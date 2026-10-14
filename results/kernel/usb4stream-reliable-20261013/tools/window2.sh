#!/bin/bash
# Window 2 part B on all six Mac Pros (about 40 min, under chain6.sh): the automatic setup, a link check of this
# branch's build, Flash-Next end to end against main's binary, then Qwen3.8-27B TP4 on two Macs. No module reloads or
# resets; links that go down stop the window.
# 0. The six ring links (ring6.py) and the chain order and node addresses (main's ring.py), from the live links.
# 1. The automatic setup on every Mac (deploy-setup.sh: tblink and the udev rule), unless DEPLOYED=1; every link up with
#    its stream.
# 2. Link check (CPU only): the raw ping-pong with three follow-up frames on every link, link-test over the streams and
#    a 2-minute soak. The window stops unless every link is on its stream, every 64 B ping-pong p50 is under 500 us and
#    no session failed.
# 3. `qwen4 validate` on the 14-token reference with main's binary (TCP) and this branch's (streams): dumps compared
#    byte for byte.
# 4. Alternated pairs, main's binary first: decode (bench --pp 512 --tg 128, two pairs), MTP (the three chat prompts,
#    --runs 1, two pairs), prefill 2048 and 8192 (one pair each), as main's suite runs them.
# 5. This branch over TCP (tblink stopped and the streams removed on every Mac, then back): decode and MTP once.
# 6. The four other Macs' leases go back, and TP4 runs on incite and its ring neighbour with the 27B model
#    (tp4-pair.sh).
# DRY=1 prints the runs and checks the binaries and inputs, touching nothing else.
# usage: chain6.sh stream-w2b 40 OWNER -- results/kernel/usb4stream-reliable-20261013/tools/window2.sh
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
MAIN=${MAIN:-trashcan-15e5457}
MINE=trashcan-stream
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/w2b-$(date +%Y%m%d-%H%M%S)}
F=/home/fairfieldt/trashcan-exp/flashnext
W=trashcan-exp/flashnext
MODEL=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
MACS="incite indite inflight alight invite enlight"
CFG=/sys/kernel/config/thunderbolt/stream
PORT=7075
mkdir -p "$OUT"
dry() { [ "${DRY:-0}" = 1 ]; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
on() { local h=$1; shift; if dry; then echo "    ssh $h: $*" >> "$OUT/steps.log"; return 0; fi; ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 "$h" "$@"; }

python3 "$here/ring6.py" > "$OUT/ring.txt" || { log "the six links do not make a ring: $(cat "$OUT/ring.txt")"; exit 1; }
R=($(awk '{print $1}' "$OUT/ring.txt"))
next_of() { awk -v h="$1" '$1 == h {print $2}' "$OUT/ring.txt"; }
addr_of() { awk -v h="$1" '$1 == h {print $3}' "$OUT/ring.txt"; }
eval "$(python3 "$repo/results/qwen4/chain-20261012/tools/ring.py")" || { log "ring.py failed"; exit 1; }
C="--gpus 0,1 --nodes $ADDRS --head-split 12 --context 32768 --batch 128 --ngram disk"

if ! dry; then
    [ -n "${TICKETS:-}" ] || { echo "run under chain6.sh (TICKETS)" >&2; exit 1; }
    ( while :; do for t in ${TICKETS//,/ }; do gpuq --on "${t%%=*}" status "${t#*=}" > /dev/null 2>&1; done; sleep 120; done ) > /dev/null 2>&1 &
    hb=$!
    # However the window ends: every background job of this script and its children, then the servers, and tblink
    # running on every Mac.
    trap 'for j in $(jobs -p); do pkill -P $j 2>/dev/null; kill $j 2>/dev/null; done
        for h in $MACS; do ssh "$h" "pkill -f \"[l]ink-test serve --listen 0.0.0.0:$PORT\"; sudo -n systemctl start tblink.service" 2>/dev/null; done' EXIT
    trap 'exit 130' INT TERM
fi

run() { # NAME BINARY DRIVER [--raw]
    local name=$1 bin=$2 driver=$3
    shift 3
    if dry; then log "  run $name ($bin): $driver"; return 0; fi
    local t0=$SECONDS
    python3 "${Q4RUN:-$repo/results/qwen4/chain-20261012/tools/q4run.py}" --tickets "$TICKETS" --binary "$bin" --name "$name" \
        --driver-host incite --nodes "$NODES" "$@" --driver "$driver" > "$OUT/$name.out" 2>&1
    log "  $name rc=$? $((SECONDS - t0)) s: $(grep -oE '"(tg|pp)_tok_s":[0-9.]+|tok/s[^,]*' "$OUT/$name.out" | head -4 | tr '\n' ' ')"
}
links_ok() { # every Mac's links up on two lanes
    local h
    dry && return 0
    for h in $MACS; do
        on "$h" /usr/local/sbin/tblink links | awk -F'\t' 'NR > 1 { n++; if ($6 != "up" || $5 != 2) bad = 1 } END { exit bad || n < 2 }' || return 1
    done
}
streams_all() { # every link up on two lanes, with its stream, on every Mac
    local h
    for h in $MACS; do
        on "$h" /usr/local/sbin/tblink links | awk -F'\t' 'NR > 1 { n++; if ($6 != "up" || $5 != 2 || $10 == "-" || $10 == "") bad = 1 } END { exit bad || n < 2 }' || return 1
    done
}
record_links() { # file
    local h
    for h in $MACS; do on "$h" /usr/local/sbin/tblink links | tail -n +2 | cut -f2,5,6,10 | sed "s/^/$h /"; done > "$1"
}

log "output $OUT, main $MAIN, this branch $MINE"
log "ring: $(awk '{print $1 "->" $3}' "$OUT/ring.txt" | tr '\n' ' '); chain incite,$NODES"
log "preflight"
for h in $MACS; do
    ssh -o ConnectTimeout=10 "$h" "test -x $F/$MAIN && test -x $F/$MINE && test -x $F/scripts/with-clocks.sh && test -f $F/scripts/latency-run.py && test -f $MODEL && test -f $F/stream-tbcheck.py" \
        || { log "$h: binaries, scripts or model missing"; exit 1; }
done
ssh incite "test -f $F/long.ids && test -f $F/long-32k.ids && test -f $MTP && ls $F/mtp-prompts/water.ids $F/mtp-prompts/csv.ids $F/mtp-prompts/transformer.ids" > /dev/null \
    || { log "incite: prompt files or MTP model missing"; exit 1; }
links_ok || { log "links not all up on two lanes: stop"; exit 1; }
log "preflight ok: $(ssh incite "md5sum $F/$MINE" | cut -c1-12) on incite"

if [ "${DEPLOYED:-0}" != 1 ]; then
    log "1. automatic stream setup on every Mac (tblink, 70-tbstream.rules)"
    if dry; then log "  deploy-setup.sh $MACS"; else "$here/deploy-setup.sh" $MACS > "$OUT/deploy-setup.txt" 2>&1; cat "$OUT/deploy-setup.txt" >> "$OUT/steps.log"; fi
fi
if ! dry; then
    t0=$SECONDS
    for t in $(seq 12); do streams_all && break; sleep 5; done
    record_links "$OUT/links-start.txt"
    streams_all || { log "links without a stream: $(tr '\n' ';' < "$OUT/links-start.txt")"; exit 1; }
    log "  every link up with its stream $((SECONDS - t0)) s after the setup: $(awk '{print $1 "-" $2 ":" $5}' "$OUT/links-start.txt" | tr '\n' ' ')"
fi

log "2. link check (CPU only): raw ping-pong with three follow-up frames, link-test over the streams, 2-minute soak"
dev_of() { awk -v h="$1" -v p="$2" '$1 == h && $2 == p {print $5}' "$OUT/links-start.txt"; }
rawpair() { # tag args...: tbcheck.py ARGS on every link at once, the server end first (`server`/`client` filled in)
    local tag=$1 h nx pids=""
    shift
    for h in "${R[@]}"; do
        nx=$(next_of "$h")
        on "$nx" "timeout -k 5 60 python3 ~/$W/stream-tbcheck.py $1 $(dev_of "$nx" "$h") server ${*:2}" >> "$OUT/probe-$tag.jsonl" 2>> "$OUT/probe-$tag.err" &
        pids="$pids $!"
    done
    sleep 0.5
    for h in "${R[@]}"; do
        nx=$(next_of "$h")
        on "$h" "timeout -k 5 60 python3 ~/$W/stream-tbcheck.py $1 $(dev_of "$h" "$nx") client ${*:2}" >> "$OUT/probe-$tag.jsonl" 2>> "$OUT/probe-$tag.err" &
        pids="$pids $!"
    done
    wait $pids
}
rawpair f3 pp 200 112 3 201
dry || python3 "$here/probe_table.py" "$OUT/probe-f3.jsonl" | tee -a "$OUT/steps.log"
# The hold against the receiving ring's position: does it follow the position mod 4 (Rx descriptors written back per
# 64-byte line of four)?
rawpair pos pos 40 202
dry || python3 -c '
import json, sys
for l in open(sys.argv[1]):
    try:
        d = json.loads(l)
    except ValueError:
        continue
    if d.get("side") == "client":
        print("  %-9s %d rounds, fillers needed by ring position mod 4 (client frames before the lone one): %s" % (d["host"], d["rounds"], d["fillers_by_position_mod_4"]))
' "$OUT/probe-pos.jsonl" | tee -a "$OUT/steps.log"
for h in $MACS; do on "$h" ": > ~/$W/stream-serve.log; pgrep -f '^[^ ]*/trashcan-stream[-a-z]* link-test serve --listen 0.0.0.0:$PORT' > /dev/null || { setsid nohup ~/$W/$MINE link-test serve --listen 0.0.0.0:$PORT >> ~/$W/stream-serve.log 2>&1 < /dev/null & }"; done
sleep 1
lt() { # tag args...: every Mac's client to the next Mac at once
    local tag=$1 h pids="" tmo=180
    shift
    case " $* " in *" soak "*) tmo=300 ;; esac
    for h in $MACS; do
        on "$h" "timeout -k 10 $tmo ~/$W/$MINE link-test run --connect $(addr_of "$h"):$PORT --label $tag-$h $*" >> "$OUT/$tag-$h.jsonl" 2>> "$OUT/$tag-$h.err" &
        pids="$pids $!"
    done
    wait $pids
}
lt stream --mode pingpong --size 64 --count 5000
lt stream --mode pingpong --size 20480 --count 2000
lt stream --mode exchange --size 40960 --count 2000
lt stream --mode bulk --size 5242880 --count 20
lt stream --mode chain --count 1000
lt soak --mode soak --seconds 120
# The stream back after a fallback (part A: on links that hold frames, a stream re-established after a fallback died
# again within ~100 ms): the server gives its stream up 2 s in, and a server stopped for 2 s.
a=${R[0]} b=${R[1]} c=${R[2]} d=${R[3]}
on "$b" "timeout -k 10 180 ~/$W/$MINE link-test run --connect $(addr_of "$b"):$PORT --label f-giveup-$b --mode chain --count 20000 --server-drops-stream 2" >> "$OUT/f-giveup-$b.jsonl" 2>> "$OUT/f-giveup-$b.err"
on "$c" "timeout -k 10 180 ~/$W/$MINE link-test run --connect $(addr_of "$c"):$PORT --label f-stop-$c --mode chain --count 30000" >> "$OUT/f-stop-$c.jsonl" 2>> "$OUT/f-stop-$c.err" &
p=$!
sleep 3
on "$d" "date +%s.%N; pkill -STOP -f '[l]ink-test serve --listen 0.0.0.0:$PORT'; sleep 2; pkill -CONT -f '[l]ink-test serve --listen 0.0.0.0:$PORT'; date +%s.%N" > "$OUT/f-stop-inject.txt" 2>&1
wait $p
for h in $MACS; do on "$h" "pkill -f '[l]ink-test serve --listen 0.0.0.0:$PORT'; cat ~/$W/stream-serve.log" > "$OUT/serve-$h.log" 2>&1; done
if ! dry; then
    python3 "$here/summary.py" "$OUT" suites | tee -a "$OUT/steps.log"
    python3 "$here/summary.py" "$OUT" soak | tee -a "$OUT/steps.log"
    python3 "$here/summary.py" "$OUT" faults | cut -c1-400 | tee -a "$OUT/steps.log"
    verdict=$(python3 -c '
import glob, json, sys
d = sys.argv[1]
bad = []
for f in sorted(glob.glob(d + "/stream-*.jsonl")) + sorted(glob.glob(d + "/soak-*.jsonl")) + sorted(glob.glob(d + "/f-*.jsonl")):
    name = f.rsplit("/", 1)[1]
    for l in open(f):
        x = json.loads(l)
        if "error" in x:
            bad.append(name + ": " + x["error"][:80])
        elif "result" in x:
            r = x["result"]
            if any(p["path"] != "stream" for p in r.get("lanes", [])):
                bad.append(name + ": not on the stream")
            t = x.get("test", {})
            if t.get("mode") == "pingpong" and t.get("size") == 64 and r["lat_us"]["p50"] > 500:
                bad.append(name + ": 64 B ping-pong p50 %.0f us" % r["lat_us"]["p50"])
print("ok" if not bad else "; ".join(bad))
' "$OUT")
    log "  link check: $verdict"
    [ "$verdict" = ok ] || { log "the link check failed: no end-to-end runs this window"; exit 1; }
fi

log "3. validation, main then this branch"
run stream-w2-val14-main $MAIN "validate results/stream-w2-val14-main $C --greedy 10"
run stream-w2-val14-stream $MINE "validate results/stream-w2-val14-stream $C --greedy 10"
if ! dry; then
    ssh incite "cd $F/results && diff -r stream-w2-val14-main stream-w2-val14-stream && echo 'dumps byte-identical'; \
        cat stream-w2-val14-main/greedy.txt | tr '\n' ' '; echo; cat stream-w2-val14-stream/greedy.txt | tr '\n' ' '" > "$OUT/val-compare.txt" 2>&1
    log "  $(tail -3 "$OUT/val-compare.txt" | tr '\n' '|')"
fi

log "4. alternated pairs, main first"
for i in 1 2; do
    run stream-w2-tg-main-$i $MAIN "bench $C --pp 512 --tg 128 --tokens-file $F/long.ids"
    run stream-w2-tg-stream-$i $MINE "bench $C --pp 512 --tg 128 --tokens-file $F/long.ids"
done
for i in 1 2; do
    for b in $MAIN $MINE; do
        side=main; [ "$b" = "$MINE" ] && side=stream
        run stream-w2-mtp-$side-$i $b "./$b qwen4 mtp spec $MODEL $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $F/mtp-prompts/water.ids $F/mtp-prompts/csv.ids $F/mtp-prompts/transformer.ids" --raw
    done
done
run stream-w2-pp2048-main $MAIN "bench $C --pp 2048 --tg 32 --tokens-file $F/long.ids"
run stream-w2-pp2048-stream $MINE "bench $C --pp 2048 --tg 32 --tokens-file $F/long.ids"
run stream-w2-pp8192-main $MAIN "bench $C --pp 8192 --tg 32 --tokens-file $F/long-32k.ids"
run stream-w2-pp8192-stream $MINE "bench $C --pp 8192 --tg 32 --tokens-file $F/long-32k.ids"

log "5. this branch over TCP: tblink stopped and the streams removed on every Mac"
for h in $MACS; do
    if dry; then log "  $h: stop tblink, remove streams"; continue; fi
    ssh "$h" "sudo -n systemctl stop tblink.service; for d in $CFG/*/trashcan; do [ -d \$d ] && sudo -n rmdir \$d && sudo -n rmdir \$(dirname \$d); done; true" >> "$OUT/tcp-toggle.txt" 2>&1
done
run stream-w2-tg-tcp-1 $MINE "bench $C --pp 512 --tg 128 --tokens-file $F/long.ids"
run stream-w2-mtp-tcp-1 $MINE "./$MINE qwen4 mtp spec $MODEL $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $F/mtp-prompts/water.ids $F/mtp-prompts/csv.ids $F/mtp-prompts/transformer.ids" --raw
for h in $MACS; do dry || ssh "$h" "sudo -n systemctl start tblink.service" >> "$OUT/tcp-toggle.txt" 2>&1; done
if ! dry; then
    for t in $(seq 12); do streams_all && break; sleep 5; done
    record_links "$OUT/links-end.txt"
    streams_all && log "  every link up with its stream again" || { log "links or streams missing after tblink restarted: $(tr '\n' ';' < "$OUT/links-end.txt")"; exit 1; }
fi

TP=$(for h in $(next_of incite) $(awk '$2 == "incite" {print $1}' "$OUT/ring.txt"); do ssh -o ConnectTimeout=10 "$h" "test -f /home/fairfieldt/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf" && { echo "$h"; break; }; done)
log "6. Qwen3.8-27B TP4 on incite + $TP: the four other Macs' leases go back first"
[ -n "$TP" ] || { log "no neighbour of incite has the 27B model"; exit 1; }
if ! dry; then
    for t in ${TICKETS//,/ }; do
        m=${t%%=*}
        [ "$m" = incite ] || [ "$m" = "$TP" ] || { gpuq --on "$m" release "${t#*=}" -r "stream-w2b: TP4 pair only from here" > /dev/null 2>&1 && log "  released $m"; }
    done
fi
PAIR=$(echo "${TICKETS:-incite=0,$TP=0}" | tr ',' '\n' | grep -E "^(incite|$TP)=" | paste -sd, -)
TICKETS=$PAIR NODE=$TP OUT=$OUT/tp4 "$here/tp4-pair.sh" 2>&1 | tee -a "$OUT/steps.log"
log "done"
echo "stream-w2b done: $OUT"
