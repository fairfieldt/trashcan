#!/bin/bash
# Decode gate, six-Mac A/B (under chain6.sh, TICKETS set): the candidate (CAND, default trashcan-dg10) against main's
# binary (BASE, default trashcan-4f36bf0), driven like results/qwen4/main-suite-20261013/tools/main-suite.sh (q4run.py,
# the chain order from the live Thunderbolt links). Each run once:
# - validation of the candidate on the 14-token reference prompt and on p100 (--greedy 10, as the main suite), bit for
#   bit against the main suite's runs of BASE (REF, default ms-4f36bf0);
# - decode after a 512-token prompt (256 steps; its decode window also gives the SMC power) and MTP on the three chat
#   prompts, one alternated pair each, and a second pair where the first differs by less than 3% (tools/dg-pair.py).
# After every run the session stops on: a failed run (a GPU that hangs is not retried); a new kernel-log line on any of
# the six Macs (amdgpu faults, PCIe AER); any link off its stream (the driver's `links:` lines, and every link-end in
# the processes' logs, transport.py), reported as a link failure with its log lines, no retry; in the candidate's runs,
# a gate that gave up or gates that went off. DRY=1 prints the runs.
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-aec14e120304a2724/results/qwen4/chain-20261012 || exit 1
CAND=${CAND:-trashcan-dg10}
BASE=${BASE:-trashcan-4f36bf0}
REF=${REF:-ms-4f36bf0}
TAG=${TAG:-dg10c}
G=/Users/fairfieldt/dev/claude-work/flashnext-20261012/gate2
L=$G/chain/$TAG
T=../decode-gate-20261014/tools
TR=../decode-kernels-20261014/tools/transport.py
W=/home/fairfieldt/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
MACS="incite indite inflight alight invite enlight"
pat='amdgpu.*(timeout|reset|fault)|CmpltTO|device inaccessible|BUG|Oops|AER|PCIe Bus Error|pci_channel_io|UnxCmplt'
eval "$(python3 tools/ring.py)" || exit 1
N=$ADDRS
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
P=$W/mtp-prompts
mkdir -p "$L"
# The kernel-log matches of every Mac, as host=count.
klog() {
    local h
    for h in $MACS; do
        echo "$h=$(ssh -o ConnectTimeout=8 "$h" "sudo -n dmesg | grep -i -c -E '$pat'")"
    done | tr '\n' ' '
}
kcheck() {
    local now h
    now=$(klog)
    if [ "$now" != "$kbase" ]; then
        echo "dg chain $TAG stopped after $1: kernel log hit (before: $kbase; now: $now)"
        for h in $MACS; do ssh -o ConnectTimeout=8 "$h" "sudo -n dmesg | grep -i -E '$pat' | tail -3" | sed "s/^/$h: /"; done
        exit 3
    fi
}
# Every link of run NAME on its stream (the driver's `links:` lines, and all 10 link-ends in the logs), else a link
# failure: the report, the link lines of every process, and the session stops.
links() {
    local name=$1 bad
    bad=$(grep -h '^links: ' "$L/$TAG-$name.err" | sed 's/^links: //; s/, /\n/g' | grep -v ' stream ' | sort -u)
    if [ -z "$bad" ] && grep -q '^links: ' "$L/$TAG-$name.err" && python3 $TR "$L" $TAG --check "$name" > /dev/null; then
        return 0
    fi
    echo "dg chain $TAG stopped after $TAG-$name: link failure"
    grep -h '^links: ' "$L/$TAG-$name.err" | sort -u | sed 's/^/    /'
    [ -n "$bad" ] && echo "$bad" | sed 's/^/    not on its stream: /'
    python3 $TR "$L" $TAG --check "$name" | sed 's/^/    /'
    grep -H -E "^link " "$L/$TAG-$name.err" "$L"/*/node-$TAG-$name-*.log 2> /dev/null | sed "s|$L/||; s/^/    /"
    exit 2
}
# In a candidate run: no gate gave up and gates stayed on; the stages with gates (12 on six Macs).
gates() {
    local name=$1 files bad
    files=("$L/$TAG-$name.err" "$L"/*/node-$TAG-$name-*.log)
    bad=$(grep -h -E "gave up after|no longer submitted ahead|not submitted ahead" "${files[@]}" 2> /dev/null)
    echo "    gates: $(cat "${files[@]}" 2> /dev/null | grep -c 'gates read') stages with gates"
    if [ -n "$bad" ]; then
        echo "dg chain $TAG stopped after $TAG-$name: gates"
        echo "$bad" | cut -c1-300 | sed 's/^/    /'
        exit 4
    fi
}
fetch() { # one run's driver outputs, dumps and clock log, and every node's log and clock log
    local name=$1 h
    rsync -a "incite:$W/results/$TAG-$name.*" "incite:$W/results/clocks/$TAG-$name.txt" "$L/" 2> /dev/null
    rsync -a "incite:$W/results/$TAG-$name" "$L/" 2> /dev/null
    for h in ${NODES//,/ }; do
        mkdir -p "$L/$h"
        rsync -a "$h:$W/results/node-$TAG-$name-$h.log" "$h:$W/results/clocks/node-$TAG-$name-$h.txt" "$L/$h/" 2> /dev/null
    done
}
run() { # name binary driver-args [q4run args]: one run, its start and end time, then its checks
    local name=$1 bin=$2 driver=$3 s rc
    shift 3
    if [ "${DRY:-0}" = 1 ]; then echo "run $TAG-$name ($bin): $driver" | cut -c1-170; return; fi
    s=$(date +%s)
    echo "$TAG-$name start $(date +%T)" >> "$L/times.txt"
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$bin" --name "$TAG-$name" --driver-host incite \
        --nodes "$NODES" "$@" --driver "$driver" > /dev/null 2>&1
    rc=$?
    echo "$TAG-$name end $(date +%T)" >> "$L/times.txt"
    fetch "$name"
    echo "$TAG-$name rc $rc $(( $(date +%s) - s )) s"
    if [ $rc -ne 0 ]; then echo "dg chain $TAG stopped: $TAG-$name failed (rc $rc)"; exit 1; fi
    kcheck "$name"
    links "$name"
    [ "$bin" = "$CAND" ] && gates "$name"
    return 0
}
valcheck() { # name: the candidate's validation against the main suite's
    [ "${DRY:-0}" = 1 ] && return
    python3 $T/dg-valcmp.py "$L/ref/$REF-$1" "$L/$TAG-$1-cand" | sed 's/^/    /'
    python3 $T/dg-valcmp.py "$L/ref/$REF-$1" "$L/$TAG-$1-cand" > /dev/null || { echo "dg chain $TAG stopped: $1 differs"; exit 5; }
}
pair() { # name driver-args-base driver-args-cand [q4run args]: one pair, a second where they differ by less than 3%
    local name=$1 base=$2 cand=$3 r
    shift 3
    for r in 1 2; do
        run "$name-base-$r" "$BASE" "$base" "$@"
        run "$name-cand-$r" "$CAND" "$cand" "$@"
        [ "${DRY:-0}" = 1 ] && continue
        echo "    $name pair $r: $(python3 $T/dg-pair.py "$L/$TAG-$name-base-$r.json" "$L/$TAG-$name-cand-$r.json")"
        python3 $T/dg-pair.py "$L/$TAG-$name-base-$r.json" "$L/$TAG-$name-cand-$r.json" > /dev/null && return 0
    done
}
echo "chain: incite,$NODES"
if [ "${DRY:-0}" != 1 ]; then
    kbase=$(klog)
    echo "kernel log matches at the start: $kbase"
    mkdir -p "$L/ref"
    rsync -a "incite:$W/results/$REF-val14" "incite:$W/results/$REF-val14.json" "incite:$W/results/$REF-val100" \
        "incite:$W/results/$REF-val100.json" "$L/ref/" || { echo "no validation runs $REF on incite"; exit 1; }
fi
run val14-cand $CAND "validate results/$TAG-val14-cand $C --greedy 10"
valcheck val14
run val100-cand $CAND "validate results/$TAG-val100-cand $C --greedy 10 --tokens-file $W/p100.ids"
valcheck val100
pair pp512 "bench $C --pp 512 --tg 256 --tokens-file $W/long.ids" "bench $C --pp 512 --tg 256 --tokens-file $W/long.ids"
spec="qwen4 mtp spec $M $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $P/water.ids $P/csv.ids $P/transformer.ids"
pair mtp "./$BASE $spec" "./$CAND $spec" --raw
[ "${DRY:-0}" = 1 ] && exit 0
echo "kernel log matches at the end: $(klog)"
echo "dg chain $TAG done"
