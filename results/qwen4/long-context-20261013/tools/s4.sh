#!/bin/bash
# Session 4 (under chain6.sh), six-Mac runs only (the single-Mac work ran before it: s4-1mac.sh, s4b/c/d). NEW =
# trashcan-long-r21 (this round, r20, rebased on main 5e48d7e), BASE = trashcan-5e48d7e (main), X = trashcan-long-r21x (NEW without
# qwen4_matf32_t, the same bits). Comparisons only report; the session always runs to the end.
#  1. validation: the 14-token reference (new); p100, the 3061-token and 8K prompts (new and base);
#  2. qwen4_matf32_t: 2048 tokens, two alternated pairs of X and NEW; chunks queued behind the running one: 8192
#     tokens, two alternated pairs of S (trashcan-long-r20s: r20 with main's order in Local::step) and NEW;
#  3. chunk size (NEW): 3072 and 4096 tokens in chunks of 64 (the default there) and 128;
#  4. benches at 512 (the standard decode bench), 8K, 16K and 30K: one alternated pair each, a second pair where
#     prefill or decode differ by under 3%;
#  5. a timeline run of NEW at 8K; 100K tokens with a Q8 K/V cache (NEW).
# DRY=1: print the run list and check the binaries and inputs on all six Macs, run nothing.
set -u
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a43eb2477e990f76f/results/qwen4/chain-20261012 || exit 1
L=$HOME/dev/claude-work/flashnext-20261012/long
CMP=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a43eb2477e990f76f/tools/compare_activations.py
REF14=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/ref/g-iq4-cpu
REF100=/Users/fairfieldt/dev/claude-work/flashnext-20261012/ref-p100
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
W=/home/fairfieldt/trashcan-exp/flashnext
TAG=${TAG:-s4}
# NEW and X: trashcan-long-r19/r19x, or the pair named in s4-new.txt (written once r20 passed its single-Mac checks).
NEW=trashcan-long-r19 BASE=trashcan-5e48d7e X=trashcan-long-r19x S=trashcan-long-r21s
[ -s "$L/s4-new.txt" ] && read -r NEW X < "$L/s4-new.txt"
echo "NEW=$NEW X=$X S=$S BASE=$BASE"
DRY=${DRY:-0}
if [ "$DRY" = 1 ]; then
    for h in incite indite inflight alight invite enlight; do
        echo "== $h: $(ssh "$h" "cd $W && md5sum $NEW $BASE $X $S 2>&1 | cut -c1-8 | tr '\n' ' '")"
    done
    ssh incite "cd $W && ls -la long.ids long-8k.ids long-32k.ids long-100k.ids p100.ids" | cut -c30-
    for f in "$CMP" "$REF14/index.tsv" "$REF100/index.tsv" tools/q4run.py; do [ -e "$f" ] && echo "ok $f" || echo "MISSING $f"; done
fi
run() { # name binary driver-args [env]
    if [ "$DRY" = 1 ]; then echo "run $1: $2 ${4:+[$4] }$3"; return; fi
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$2" --name "$1" --driver-host incite \
        --nodes indite,inflight,alight,invite,enlight --env "${4:-}" --driver "$3" > /dev/null 2>&1
    echo "$1 rc=$? $(ssh incite "head -c 160 $W/results/$1.json")"
}
cmpval() { # A B LABEL: compare_activations of two validate directories (both must exist)
    [ "$DRY" = 1 ] && return
    for d in "$1" "$2"; do [ -e "$d/index.tsv" ] || { echo "== $3: missing $d"; return; }; done
    echo "== $3"; python3 "$CMP" "$1" "$2" 2>&1 | grep -i "logits\|worst"
}
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --ngram disk --batch 128"
# 1. validation
run long-$TAG-val14-new $NEW "validate results/long-$TAG-val14-new $C --greedy 10"
for b in new base; do
    bin=$NEW; [ $b = base ] && bin=$BASE
    run long-$TAG-val100-$b $bin "validate results/long-$TAG-val100-$b $C --greedy 10 --tokens-file $W/p100.ids"
    run long-$TAG-val3061-$b $bin "validate results/long-$TAG-val3061-$b $C --greedy 10 --tokens-file $W/long.ids"
    run long-$TAG-val8k-$b $bin "validate results/long-$TAG-val8k-$b $C --greedy 10 --tokens-file $W/long-8k.ids"
done
if [ "$DRY" != 1 ]; then
    mkdir -p "$L/val"
    rsync -a "incite:$W/results/long-$TAG-val*" "$L/val/"
    V=$L/val/long-$TAG
    cmpval "$V-val14-new" "$REF14" "val14 new vs the CPU reference"
    cmpval "$V-val100-new" "$REF100" "p100 new vs the CPU reference"
    cmpval "$V-val100-base" "$REF100" "p100 base vs the CPU reference"
    for v in val100 val3061 val8k; do
        cmpval "$V-$v-new" "$V-$v-base" "$v new vs base"
        if [ -e "$V-$v-new/greedy.txt" ] && [ -e "$V-$v-base/greedy.txt" ]; then
            cmp -s "$V-$v-new/greedy.txt" "$V-$v-base/greedy.txt" && echo "$v: greedy tokens identical" \
                || echo "CHECK: $v greedy tokens differ"
        else echo "CHECK: $v greedy.txt missing"; fi
    done
    for d in "$L"/val/long-$TAG-val*; do [ -e "$d/greedy.txt" ] && echo "$(basename "$d") greedy: $(tr '\n' ' ' < "$d/greedy.txt")"; done
fi
# 2. qwen4_matf32_t (the indexer and DeltaNet ab projections of prefill chunks): two alternated pairs
for r in 1 2; do
    order="x new"; [ $r = 2 ] && order="new x"
    for b in $order; do
        bin=$NEW; [ $b = x ] && bin=$X
        run long-$TAG-pp2048-$b-$r $bin "bench $C --pp 2048 --tg 16 --tokens-file $W/long-32k.ids"
    done
done
# 2b. a prompt chunk queued behind the running one (Local::step): two alternated pairs of S (main's order: wait for the
# running chunk, then submit the next) and NEW at 8192 tokens
for r in 1 2; do
    order="s new"; [ $r = 2 ] && order="new s"
    for b in $order; do
        bin=$NEW; [ $b = s ] && bin=$S
        run long-$TAG-pp8192q-$b-$r $bin "bench $C --pp 8192 --tg 16 --tokens-file $W/long-32k.ids"
    done
done
# 3. chunk size: 3072 and 4096 tokens in chunks of 64 (the default there) and of 128
for pp in 3072 4096; do
    run long-$TAG-pp$pp-c64 $NEW "bench $C --pp $pp --tg 16 --tokens-file $W/long-32k.ids"
    run long-$TAG-pp$pp-c128 $NEW "bench $C --pp $pp --tg 16 --tokens-file $W/long-32k.ids" "TRASHCAN_Q4_CHUNK=128"
done
# 4. benches: one alternated pair per size, a second where prefill or decode differ by under 3%
rates() { ssh incite "python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r[\"pp_tok_s\"], r[\"tg_tok_s\"])' $W/results/$1.json" 2>/dev/null; }
close() { # NAME_A NAME_B: exit 0 when prefill or decode rates differ by under 3% (or a result is missing)
    local a b
    a=$(rates "$1") b=$(rates "$2")
    [ -n "$a" ] && [ -n "$b" ] || return 0
    python3 -c "import sys; a=list(map(float,sys.argv[1].split())); b=list(map(float,sys.argv[2].split())); \
        sys.exit(0 if min(abs(x/y-1) for x,y in zip(a,b))<0.03 else 1)" "$a" "$b"
}
# (512: the standard decode bench, --tg 128 on long.ids; the decode attention kernels serve every context)
i=0
for pp in 512 8192 16384 30000; do
    order="new base"; [ $((i % 2)) = 1 ] && order="base new"; i=$((i + 1))
    args="--pp $pp --tg 32 --tokens-file $W/long-32k.ids"; [ $pp = 512 ] && args="--pp 512 --tg 128 --tokens-file $W/long.ids"
    for b in $order; do
        bin=$NEW; [ $b = base ] && bin=$BASE
        run long-$TAG-pp$pp-$b-1 $bin "bench $C $args"
    done
    if [ "$DRY" = 1 ] || close long-$TAG-pp$pp-new-1 long-$TAG-pp$pp-base-1; then
        [ "$DRY" = 1 ] && echo "(only if prefill or decode differ by under 3%:)"
        for b in $order; do
            bin=$NEW; [ $b = base ] && bin=$BASE
            run long-$TAG-pp$pp-$b-2 $bin "bench $C $args"
        done
    fi
done
# 5. timeline; 100K tokens with a Q8 K/V cache (G-iq4's longest context: 114,688 positions)
run long-$TAG-tl-pp8192-new $NEW "bench $C --pp 8192 --tg 32 --tokens-file $W/long-32k.ids" \
    "TRASHCAN_Q4_TIMELINE=1 TRASHCAN_TP_PROFILE=1"
C100="--gpus 0,1 --nodes $N --head-split 12 --context 114688 --kv q8 --ngram disk --batch 128"
run long-$TAG-pp100k-new $NEW "bench $C100 --pp 100000 --tg 16 --tokens-file $W/long-100k.ids"
echo "$TAG done"
