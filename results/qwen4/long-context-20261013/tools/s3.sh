#!/bin/bash
# Session 3 (under chain6.sh): NEW (trashcan-long-r10 = c1da0cd: session 2's r8 plus the decode merge kernel and two
# submission slots per stage) against main 4c1346d.
#  0. kernel check on inflight as in session 2 (plus the block scores' bits), stage-bench profiles;
#  1. validation: 3061-token and 8K prompts (new): the same greedy tokens as session 2's base, and the prompt's dumps
#     the same bits as session 2's r8 (the slots change only when the host submits);
#  2. chunk size: 3072 and 4096 tokens in chunks of 64 and of 128 (new; the bands pick 64 at both);
#  3. benches at 8K, 16K and 30K, alternating new and base (second round, new first); a timeline run of new at 8K.
set -u
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a43eb2477e990f76f/results/qwen4/chain-20261012 || exit 1
L=$HOME/dev/claude-work/flashnext-20261012/long
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
W=/home/fairfieldt/trashcan-exp/flashnext
TAG=${TAG:-s3}
NEW=${NEW:-trashcan-long-r10}
BASE=trashcan-4c1346d
ticket() { echo "$TICKETS" | tr ',' '\n' | sed -n "s/^$1=//p"; }
# 0. kernel check
ssh inflight "cd $W && gpuq run $(ticket inflight) -- ./long-unit.sh $NEW $TAG 2049 2052 2101 8192 8193 8196 30000 30081" || exit 1
mkdir -p "$L/dumps" "$L/sb"
rsync -a "inflight:$W/results/long-dump-$TAG-*" "$L/dumps/"
ok=1
for n in 2049 2052 2101 8192 8193 8196 30000 30081; do
    out=$(python3 "$L/dumpcmp.py" "$L/dumps/long-dump-$TAG-$n-new" "$L/dumps/long-dump-$TAG-$n-old")
    echo "== $n"; echo "$out" | grep -v identical | grep -v '^indexer_scores'
    echo "$out" | grep -q '^indexer_sel-3 .* 0 rows differ (0 as sets)' || { echo "$n: layer 3 selection differs or missing"; ok=0; }
    echo "$out" | grep -E '^[^ ]*-[012](#2)? |^indexer_q-3 ' | grep -v identical && { echo "$n: layers 0-2 differ"; ok=0; }
    echo "$out" | awk '/^attn_pregate/ && $3+0 > 1e-4 {bad=1} END {exit bad}' || { echo "$n: attention output off"; ok=0; }
    echo "$out" | awk '/^l_last-7 / && $3+0 > 1e-3 {bad=1} END {exit bad}' || { echo "$n: stage output off"; ok=0; }
    python3 "$L/scorecmp.py" "$L/dumps/long-dump-$TAG-$n-new" "$L/dumps/long-dump-$TAG-$n-old" $n
done
[ $ok = 1 ] || { echo "kernel check failed"; exit 1; }
ssh inflight "cd $W && gpuq run $(ticket inflight) -- ./long-sb.sh $NEW $TAG 8320 30080"
ssh inflight "cd $W && EXTRA=TRASHCAN_Q4_SCORE=lds gpuq run $(ticket inflight) -- ./long-sb.sh $NEW $TAG-scorelds 30080"
rsync -a "inflight:$W/results/long-sb-$TAG-*" "$L/sb/"
run() { # name binary driver-args [env]
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$2" --name "$1" --driver-host incite \
        --nodes indite,inflight,alight,invite,enlight --env "${4:-}" --driver "$3" > /dev/null 2>&1
    echo "$1 rc=$? $(ssh incite "head -c 200 $W/results/$1.json")"
}
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --ngram disk"
# 1. validation
for v in val3061:long.ids val8k:long-8k.ids; do
    run long-$TAG-${v%%:*}-new $NEW "validate results/long-$TAG-${v%%:*}-new $C --batch 128 --greedy 10 --tokens-file $W/${v#*:}"
    ssh incite "cd $W/results && cmp long-$TAG-${v%%:*}-new/greedy.txt long-s2-${v%%:*}-base/greedy.txt" \
        || { echo "${v%%:*}: greedy tokens differ from base"; exit 1; }
done
echo "validation: greedy tokens identical to base"
# 2. chunk size (TRASHCAN_Q4_CHUNK forces it; the bands pick 64 at both lengths)
for pp in 3072 4096; do
    for b in 64 128; do
        run long-$TAG-pp$pp-c$b $NEW "bench $C --batch 128 --pp $pp --tg 16 --tokens-file $W/long-32k.ids" \
            "TRASHCAN_Q4_CHUNK=$b"
    done
done
# 3. benches, second round
for pp in 8192 16384 30000; do
    for b in new base; do
        bin=$NEW; [ $b = base ] && bin=$BASE
        run long-$TAG-pp$pp-$b-2 $bin "bench $C --batch 128 --pp $pp --tg 32 --tokens-file $W/long-32k.ids"
    done
done
run long-$TAG-tl-pp8192-new $NEW "bench $C --batch 128 --pp 8192 --tg 32 --tokens-file $W/long-32k.ids" \
    "TRASHCAN_Q4_TIMELINE=1 TRASHCAN_TP_PROFILE=1"
echo "$TAG done"
