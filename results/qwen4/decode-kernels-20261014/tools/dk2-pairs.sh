#!/bin/bash
# dk2 six-Mac pairs after the stream fix (under chain6.sh, TICKETS set): c2 (CAND) against main (BASE), driven as
# dk2-chain.sh: decode after a 512-token prompt (one pair), prefill at 2048 and 8192 tokens and decode after them (one
# pair each), arms alternated. Each run happens once: after every run its logs are fetched and tools/transport.py
# checks that all 10 link-ends ran on their streams (a node's upstream link closed by the other side at its exit counts
# only under transport.py's "closed at end" rule, and is printed); if one did not, the session stops and prints the
# run, the link and its log lines. Stops at the first failed run too. DRY=1 prints the runs.
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a70283bc249c19df0/results/qwen4/chain-20261012 || exit 1
CAND=${CAND:-trashcan-dk2-c2}
BASE=${BASE:-trashcan-9baedca}
TAG=${TAG:-dk2c2p}
DK=/Users/fairfieldt/dev/claude-work/flashnext-20261012/dk2
L=$DK/chain/$TAG
W=/home/fairfieldt/trashcan-exp/flashnext
eval "$(python3 tools/ring.py)" || exit 1
N=$ADDRS
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
fetch() { # one run's driver outputs and clock log, and every node's log and clock log
    local name=$1 h
    mkdir -p "$L"
    rsync -a "incite:$W/results/$TAG-$name.*" "incite:$W/results/clocks/$TAG-$name.txt" "$L/" 2> /dev/null
    for h in ${NODES//,/ }; do
        mkdir -p "$L/$h"
        rsync -a "$h:$W/results/node-$TAG-$name-$h.log" "$h:$W/results/clocks/node-$TAG-$name-$h.txt" "$L/$h/" 2> /dev/null
    done
}
run() { # name binary driver-args
    local name=$1 bin=$2 driver=$3
    if [ "${DRY:-0}" = 1 ]; then echo "run $TAG-$name ($bin): $driver" | cut -c1-160; return; fi
    local s=$(date +%s)
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$bin" --name "$TAG-$name" --driver-host incite \
        --nodes "$NODES" --driver "$driver" > /dev/null 2>&1
    local rc=$?
    fetch "$name"
    echo $rc > "$L/$TAG-$name.rc"
    echo "$TAG-$name rc $rc $(( $(date +%s) - s )) s"
    if [ $rc -ne 0 ]; then echo "dk2 pairs $TAG stopped: $TAG-$name failed (rc $rc)"; exit 1; fi
    if ! python3 $DK/tools/transport.py "$L" $TAG --check "$name"; then
        echo "dk2 pairs $TAG stopped: $TAG-$name ran a link-end off its stream"
        exit 1
    fi
}
echo "chain: incite,$NODES"
if [ "${DRY:-0}" != 1 ]; then mkdir -p "$L"; echo "incite,$NODES" > "$L/chain.txt"; fi
run pp512-base $BASE "bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
run pp512-cand $CAND "bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
run pp2048-base $BASE "bench $C --pp 2048 --tg 32 --tokens-file $W/long.ids"
run pp2048-cand $CAND "bench $C --pp 2048 --tg 32 --tokens-file $W/long.ids"
run pp8192-base $BASE "bench $C --pp 8192 --tg 32 --tokens-file $W/long-32k.ids"
run pp8192-cand $CAND "bench $C --pp 8192 --tg 32 --tokens-file $W/long-32k.ids"
echo "dk2 pairs $TAG done"
