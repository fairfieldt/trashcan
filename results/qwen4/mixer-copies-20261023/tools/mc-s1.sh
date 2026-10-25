#!/bin/bash
# Mixer copies, session 1 (one Mac, GPU 0, under a gpuq lease; run in ~/trashcan-exp/flashnext): main (BASE) against
# the candidate without the Q8_0 mixers' separate down copies and f32 write weights (CAND).
# 1. Allocation of layers 0-3 with the first twelfth of the head at 32K, 128- and 256-token chunks (the six-Mac first
#    stage; measured before: 5830.4 MiB at 128).
# 2. Stage dumps of layers 4-7, both binaries: the 14-token prompt one token at a time (fused mixer kernels) and in
#    chunks of 4, p100 in chunks of 16 (Q8_0 lane GEMM) and of 128 (qwen4_q8_gemm), 2101 tokens one at a time.
# 3. One alternated timing pair (decode after 512 tokens, one 128-token prefill chunk); no kernel changed.
set -uo pipefail
cd ~/trashcan-exp/flashnext
BASE=${BASE:-trashcan-b7f4979}
CAND=${CAND:-trashcan-mixc1}
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
R=results/mc/s1
mkdir -p $R
for bin in $BASE $CAND; do sha256sum $bin; done | tee $R/sha256.txt
for b in 128 256; do
    for bin in $BASE $CAND; do
        name=alloc-${bin#trashcan-}-b$b
        ./$bin qwen4 stage-bench $M --ranges 0:4 --gpus 0 --context 32768 --batch $b --head-rows 0:21504 \
            --start 64 --decode 4 --prefill 1 > $R/$name.json 2> $R/$name.err
        echo "$name rc=$? $(grep -h 'MiB allocated' $R/$name.err | cut -c1-140)"
    done
done
dump() { # name binary batch ids
    rm -rf $R/$1
    ./$2 qwen4 stage-bench $M --ranges 4:8 --gpus 0 --context 32768 --batch $3 --tokens-file $4 --out $R/$1 \
        > $R/$1.json 2> $R/$1.err
    echo "$1 rc=$?"
}
for d in b1:1:dec-ref14.ids b4:4:dec-ref14.ids p16:16:dec-p100.ids p128:128:dec-p100.ids l2101:1:long-2101.ids; do
    IFS=: read tag b ids <<< "$d"
    dump dump-base-$tag $BASE $b $ids
    dump dump-cand-$tag $CAND $b $ids
done
SB="qwen4 stage-bench $M --ranges 4:8 --gpus 0 --context 32768 --start 512 --batch 128 --decode 64 --prefill 1"
for r in base:$BASE cand:$CAND base2:$BASE cand2:$CAND; do
    scripts/with-clocks.sh $R/time-${r%%:*}.clk ./${r#*:} $SB > $R/time-${r%%:*}.json 2> $R/time-${r%%:*}.err
    echo "time-${r%%:*} rc=$? $(grep -h clocks $R/time-${r%%:*}.err | cut -c1-140)"
done
echo "mc-s1 done"
