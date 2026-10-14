#!/bin/bash
# dk2 session 7 (one Mac, GPU 0, under a gpuq lease): v5 (gate/up 2 + 2 rows for one token, decode attention queries
# through LDS, QSA post-projection kernels in one group, gdn state loaded unconditionally) against main: decode of
# layers 4-7 after 512 and after 4096 tokens, alternated, profiled; dumps (14 tokens one at a time and in chunks of 4,
# 100 tokens in chunks of 16, 2101 tokens one at a time: sparse attention past 2051).
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
R=results/dk2/s7
mkdir -p $R
SB="qwen4 stage-bench $M --ranges 4:8 --gpus 0 --context 32768 --decode 64 --batch 128 --prefill 0"
run() {
    name=$1 bin=$2 start=$3
    shift 3
    env "$@" scripts/with-clocks.sh $R/$name.clk ./$bin $SB --start $start > $R/$name.json 2> $R/$name.err
    echo "$name rc=$? $(grep clocks $R/$name.err | cut -c1-160)"
}
dump() {
    name=$1 bin=$2 b=$3 ids=$4
    rm -rf $R/$name
    ./$bin qwen4 stage-bench $M --ranges 4:8 --gpus 0 --context 32768 --batch $b --tokens-file $ids \
        --out $R/$name > $R/$name.json 2> $R/$name.err
    echo "$name rc=$?"
}
run main-512 trashcan-15e5457 512
run v5-512 trashcan-dk2-v5 512
run main-512b trashcan-15e5457 512
run v5-512b trashcan-dk2-v5 512
run main-512p trashcan-15e5457 512 TRASHCAN_Q4_PROFILE=1
run v5-512p trashcan-dk2-v5 512 TRASHCAN_Q4_PROFILE=1
run main-4096p trashcan-15e5457 4096 TRASHCAN_Q4_PROFILE=1
run v5-4096p trashcan-dk2-v5 4096 TRASHCAN_Q4_PROFILE=1
dump dump-main-b1 trashcan-15e5457 1 dec-ref14.ids
dump dump-v5-b1 trashcan-dk2-v5 1 dec-ref14.ids
dump dump-main-b4 trashcan-15e5457 4 dec-ref14.ids
dump dump-v5-b4 trashcan-dk2-v5 4 dec-ref14.ids
dump dump-main-p16 trashcan-15e5457 16 dec-p100.ids
dump dump-v5-p16 trashcan-dk2-v5 16 dec-p100.ids
dump dump-main-l2101 trashcan-15e5457 1 long-2101.ids
dump dump-v5-l2101 trashcan-dk2-v5 1 long-2101.ids
echo "dk2-s7 done"
