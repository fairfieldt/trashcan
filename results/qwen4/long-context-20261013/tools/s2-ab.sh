#!/bin/bash
# Session 2 (under chain6.sh): the long-context round (trashcan-long = r7) against main 4c1346d.
#  0. kernel check on inflight (layers 0-3 on GPU 0, 4-7 on GPU 1, every token's rows of the last chunk, the kernels
#     before this round (TRASHCAN_Q4_QSA=old TRASHCAN_Q4_COPY_OLD=1) against after, in the same binary); stops unless
#     every selection is identical and attention outputs are within 1e-4; then stage-bench profiles;
#  1. six-Mac validation: the 14-token reference prompt (new), the 3061-token long.ids and an 8K prompt (new, base);
#     stops unless new and base give the same greedy tokens;
#  2. benches at 8K, 16K and 30K real-text tokens, alternating base and new; a timeline run of new at 8K.
set -u
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a43eb2477e990f76f/results/qwen4/chain-20261012 || exit 1
L=$HOME/dev/claude-work/flashnext-20261012/long
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
W=/home/fairfieldt/trashcan-exp/flashnext
TAG=${TAG:-s2}
NEW=${NEW:-trashcan-long}
BASE=trashcan-4c1346d
ticket() { echo "$TICKETS" | tr ',' '\n' | sed -n "s/^$1=//p"; }
# 0. kernel check
ssh inflight "cd $W && gpuq run $(ticket inflight) -- ./long-unit.sh $NEW $TAG 2049 2052 2101 8192 8193 8196 30000 30081" || exit 1
mkdir -p "$L/dumps" "$L/sb"
rsync -a "inflight:$W/results/long-dump-$TAG-*" "$L/dumps/"
ok=1
for n in 2049 2052 2101 8192 8193 8196 30000 30081; do
    out=$(python3 "$L/dumpcmp.py" "$L/dumps/long-dump-$TAG-$n-new" "$L/dumps/long-dump-$TAG-$n-old")
    echo "== $n"; echo "$out" | grep -v identical
    # Layer 3 sees the same inputs in both runs: its selections must be identical, and so must every tensor of
    # layers 0-2 and layer 3's indexer queries (the f32 projections keep their bits). Layer 7's inputs differ slightly.
    echo "$out" | grep -q '^indexer_sel-3 .* 0 rows differ (0 as sets)' || { echo "$n: layer 3 selection differs or missing"; ok=0; }
    echo "$out" | grep -E '^[^ ]*-[012](#2)? |^indexer_q-3 ' | grep -v identical && { echo "$n: layers 0-2 differ"; ok=0; }
    echo "$out" | awk '/^attn_pregate/ && $3+0 > 1e-4 {bad=1} END {exit bad}' || { echo "$n: attention output off"; ok=0; }
    echo "$out" | awk '/^l_last-7 / && $3+0 > 1e-3 {bad=1} END {exit bad}' || { echo "$n: stage output off"; ok=0; }
done
[ $ok = 1 ] || { echo "kernel check failed"; exit 1; }
ssh inflight "cd $W && gpuq run $(ticket inflight) -- ./long-sb.sh $NEW $TAG 8320 30080"
ssh inflight "cd $W && EXTRA=TRASHCAN_Q4_COPY_OLD=1 gpuq run $(ticket inflight) -- ./long-sb.sh $NEW $TAG-copyold 8320"
ssh inflight "cd $W && EXTRA=TRASHCAN_Q4_SEL=256 gpuq run $(ticket inflight) -- ./long-sb.sh $NEW $TAG-sel256 8320 30080"
rsync -a "inflight:$W/results/long-sb-$TAG-*" "$L/sb/"
run() { # name binary driver-args [env]
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$2" --name "$1" --driver-host incite \
        --nodes indite,inflight,alight,invite,enlight --env "${4:-}" --driver "$3" > /dev/null 2>&1
    echo "$1 rc=$? $(ssh incite "head -c 200 $W/results/$1.json")"
}
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
# 1. validation
run long-$TAG-val14-new $NEW "validate results/long-$TAG-val14-new $C --greedy 10"
for b in new base; do
    bin=$NEW; [ $b = base ] && bin=$BASE
    run long-$TAG-val3061-$b $bin "validate results/long-$TAG-val3061-$b $C --greedy 10 --tokens-file $W/long.ids"
    run long-$TAG-val8k-$b $bin "validate results/long-$TAG-val8k-$b $C --greedy 10 --tokens-file $W/long-8k.ids"
done
for v in val3061 val8k; do
    ssh incite "cd $W/results && cmp long-$TAG-$v-new/greedy.txt long-$TAG-$v-base/greedy.txt" \
        || { echo "$v: greedy tokens differ"; exit 1; }
done
echo "validation: greedy tokens identical"
# 2. benches
for pp in 8192 16384 30000; do
    for b in base new; do
        bin=$NEW; [ $b = base ] && bin=$BASE
        run long-$TAG-pp$pp-$b-1 $bin "bench $C --pp $pp --tg 32 --tokens-file $W/long-32k.ids"
    done
done
run long-$TAG-tl-pp8192-new $NEW "bench $C --pp 8192 --tg 32 --tokens-file $W/long-32k.ids" \
    "TRASHCAN_Q4_TIMELINE=1 TRASHCAN_TP_PROFILE=1"
echo "$TAG done"
