#!/bin/bash
# Decode gate, six-Mac A/B (under chain6.sh, TICKETS set): the candidate (CAND, default trashcan-dg8) against main
# (BASE, default trashcan-c43be4a), driven like results/qwen4/decode-kernels-20261014/tools/dk2-chain.sh (q4run.py,
# the chain order from the live Thunderbolt links). Each run once:
# - validation on the 14-token reference prompt and on p100, both arms (dumps and 32 greedy tokens compared after);
# - decode after a 512-token prompt (128 steps) and MTP on the three chat prompts, alternated pairs (two each);
# - a long decode per arm (1024 steps) for the decode window's SMC power (netconsole telemetry, power.py).
# After every run: all 10 link-ends must have run on their streams (transport.py --check), else the session stops as a
# link failure (no retry); the kernel log of all six Macs (amdgpu faults, PCIe AER) against the session's start: the
# first new line, or a failed run (a GPU that hangs is not retried), stops the session too. DRY=1 prints the runs.
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-aec14e120304a2724/results/qwen4/chain-20261012 || exit 1
CAND=${CAND:-trashcan-dg8}
BASE=${BASE:-trashcan-c43be4a}
TAG=${TAG:-dg8c}
G=/Users/fairfieldt/dev/claude-work/flashnext-20261012/gate2
L=$G/chain/$TAG
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
# Every link-end of run NAME on its stream, else a link failure: which link-ends, their log lines, and the session stops.
links() {
    local name=$1
    python3 $TR "$L" $TAG --check "$name" > /dev/null && return 0
    echo "dg chain $TAG stopped after $TAG-$name: link failure"
    python3 $TR "$L" $TAG --check "$name" | sed 's/^/    /'
    python3 $TR "$L" $TAG | grep -E "^$name +[0-9]" | sed 's/^/    /'
    grep -H -E "^link " "$L/$TAG-$name.err" "$L"/*/node-$TAG-$name-*.log 2> /dev/null | sed "s|$L/||; s/^/    /"
    exit 2
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
    echo "$TAG-$name rc $rc $(( $(date +%s) - s )) s; $(python3 $TR "$L" $TAG --check "$name")"
    if [ $rc -ne 0 ]; then echo "dg chain $TAG stopped: $TAG-$name failed (rc $rc)"; exit 1; fi
    kcheck "$name"
    links "$name"
}
echo "chain: incite,$NODES"
[ "${DRY:-0}" = 1 ] || { kbase=$(klog); echo "kernel log matches at the start: $kbase"; }
run val14-base $BASE "validate results/$TAG-val14-base $C --greedy 32"
run val14-cand $CAND "validate results/$TAG-val14-cand $C --greedy 32"
run val100-base $BASE "validate results/$TAG-val100-base $C --greedy 32 --tokens-file $W/p100.ids"
run val100-cand $CAND "validate results/$TAG-val100-cand $C --greedy 32 --tokens-file $W/p100.ids"
for r in 1 2; do
    run pp512-base-$r $BASE "bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
    run pp512-cand-$r $CAND "bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
done
for r in 1 2; do
    run mtp-base-$r $BASE "./$BASE qwen4 mtp spec $M $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $P/water.ids $P/csv.ids $P/transformer.ids" --raw
    run mtp-cand-$r $CAND "./$CAND qwen4 mtp spec $M $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $P/water.ids $P/csv.ids $P/transformer.ids" --raw
done
run power-base $BASE "bench $C --pp 512 --tg 1024 --tokens-file $W/long.ids"
run power-cand $CAND "bench $C --pp 512 --tg 1024 --tokens-file $W/long.ids"
[ "${DRY:-0}" = 1 ] && exit 0
echo "kernel log matches at the end: $(klog)"
echo "dg chain $TAG done"
