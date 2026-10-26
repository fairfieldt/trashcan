#!/bin/bash
# Cold-cache greedy validation, six Macs (under chain6.sh, TICKETS set): N times, drop every node's page cache, then
# `qwen4 validate` on the 14-token reference prompt with BIN; print each run's greedy tokens and link events.
cd $HOME/dev/claude-work/trashcan-release-20261026/fallback/lab/results/qwen4/chain-20261012 || exit 1
N=${1:-6}; BIN=${BIN:-trashcan-fb2}; TAG=${TAG:-fbc}; COLD=${COLD:-1}
W=/home/fairfieldt/trashcan-exp/flashnext
eval "$(python3 tools/ring.py)" || exit 1
C="--gpus 0,1 --nodes $ADDRS --head-split 12 --context 32768 --ngram disk"
REF="[9419, 0, 2500, 628, 353, 1438, 488, 3242, 30, 248046]"
for i in $(seq $N); do
    if [ "$COLD" = 1 ]; then
        for h in incite ${NODES//,/ }; do ssh $h "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null" & done; wait
    fi
    name=$TAG-$i
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$BIN" --name "$name" --driver-host incite --nodes "$NODES" \
        --driver "validate results/$name $C --batch 128 --greedy 10" > /dev/null 2>&1
    rc=$?
    g=$(ssh incite "grep -h greedy $W/results/$name.json 2>/dev/null | head -1")
    ok=$([ "$g" = "greedy: $REF" ] && echo SAME || echo DIFFERENT)
    echo "$name rc $rc $ok $g"
    for h in incite ${NODES//,/ }; do
        ssh $h "grep -hE 'stream lost|not up within|a turn took|no turn for|not opened|continuing over' $W/results/$name.err $W/results/node-$name-$h.log 2>/dev/null" | sed "s/^/    $h: /"
    done
done
echo "cold-val $TAG done"
