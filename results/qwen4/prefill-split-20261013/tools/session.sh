#!/bin/bash
# First six-Mac session after the 2026-10-13 reboots: main's suite on adf8cfd (the load-time merge), then the prefill
# out-projection split fix (trashcan-split8: K split in 8 parts when the partial sums fit, else 4, for every batch over 8
# tokens; trashcan-split4: always 4) against main at the prompt sizes whose chunks are 16-64 tokens. Under chain6.sh.
# DRY=1 lists the runs.
cd /Users/fairfieldt/dev/trashcan/results/qwen4/chain-20261012 || exit 1
BIN=trashcan-adf8cfd /Users/fairfieldt/dev/trashcan/results/qwen4/main-suite-20261013/tools/main-suite.sh
W=/home/fairfieldt/trashcan-exp/flashnext
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
run() { # name binary driver-args
    if [ "${DRY:-0}" = 1 ]; then echo "run $1 ($2): $3" | cut -c1-150; return; fi
    local s=$(date +%s)
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$2" --name "$1" --driver-host incite \
        --nodes indite,inflight,alight,invite,enlight --driver "$3" > /dev/null 2>&1
    echo "$1 rc $? $(( $(date +%s) - s )) s $(ssh incite "grep -o '\"pp_tok_s\":[0-9.]*' $W/results/$1.json" 2>/dev/null)"
}
run ps-val14-split8 trashcan-split8 "validate results/ps-val14-split8 $C --greedy 10"
run ps-val100-split8 trashcan-split8 "validate results/ps-val100-split8 $C --greedy 10 --tokens-file $W/p100.ids"
for pp in 512 1024 2048; do
    for b in adf8cfd split8 split4; do
        run ps-pp$pp-$b trashcan-$b "bench $C --pp $pp --tg 16 --tokens-file $W/long.ids"
    done
done
[ "${DRY:-0}" = 1 ] && exit 0
mkdir -p /Users/fairfieldt/dev/trashcan/results/qwen4/prefill-split-20261013/runs
rsync -a "incite:$W/results/ps-*" /Users/fairfieldt/dev/trashcan/results/qwen4/prefill-split-20261013/runs/
echo "prefill split session done"
