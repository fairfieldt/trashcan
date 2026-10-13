#!/bin/bash
# Chain setup with every machine loading at once (trashcan-setpar) against main 4c1346d (one machine after another):
# wall time of whole driver runs (validate on the 14-token reference, a short bench), alternating. Under chain6.sh.
cd /Users/fairfieldt/dev/trashcan/results/qwen4/chain-20261012 || exit 1
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
W=/home/fairfieldt/trashcan-exp/flashnext
run() { s=$(date +%s); python3 tools/q4run.py --tickets "$TICKETS" --binary "$2" --name "$1" --driver-host incite \
    --nodes indite,inflight,alight,invite,enlight --driver "$3" > /dev/null 2>&1; echo "$1 $(( $(date +%s) - s )) s"; }
for r in 1 2; do
  run sp-val14-4c1346d-$r trashcan-4c1346d "validate results/sp-val14-4c1346d-$r $C --greedy 10"
  run sp-val14-setpar-$r trashcan-setpar "validate results/sp-val14-setpar-$r $C --greedy 10"
done
run sp-bench-setpar trashcan-setpar "bench $C --pp 512 --tg 64 --tokens-file $W/long.ids"
ssh incite 'cd ~/trashcan-exp/flashnext/results; for f in sp-val14-*.json sp-bench-setpar.json; do echo "$f: $(head -c 120 $f)"; done'
echo "setup check done"
