#!/bin/bash
# dk2 session 6 (one Mac, GPU 0, under a gpuq lease): v4 (gate/up 2 + 2 rows for one token) with the mixer variants
# k5 (first half, one step per wave) and upl (second half, bottleneck values through LDS), profiled; dumps against
# main's.
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
R=results/dk2/s6
mkdir -p $R
SB="qwen4 stage-bench $M --ranges 4:8 --gpus 0 --context 32768 --start 512 --decode 64 --batch 128 --prefill 0"
run() {
    name=$1 bin=$2
    shift 2
    env "$@" scripts/with-clocks.sh $R/$name.clk ./$bin $SB > $R/$name.json 2> $R/$name.err
    echo "$name rc=$? $(grep clocks $R/$name.err | cut -c1-160)"
}
dump() {
    name=$1 bin=$2 b=$3 ids=$4
    shift 4
    rm -rf $R/$name
    env "$@" ./$bin qwen4 stage-bench $M --ranges 4:8 --gpus 0 --context 32768 --batch $b --tokens-file $ids \
        --out $R/$name > $R/$name.json 2> $R/$name.err
    echo "$name rc=$?"
}
run v4-p trashcan-dk2-v4 TRASHCAN_Q4_PROFILE=1
run k5-p trashcan-dk2-v4 TRASHCAN_Q4_PROFILE=1 TRASHCAN_Q4_VAR=k5
run upl-p trashcan-dk2-v4 TRASHCAN_Q4_PROFILE=1 TRASHCAN_Q4_VAR=upl
run both-p trashcan-dk2-v4 TRASHCAN_Q4_PROFILE=1 TRASHCAN_Q4_VAR=k5,upl
run main-p trashcan-15e5457 TRASHCAN_Q4_PROFILE=1
dump dump-main-b1 trashcan-15e5457 1 dec-ref14.ids
dump dump-both-b1 trashcan-dk2-v4 1 dec-ref14.ids TRASHCAN_Q4_VAR=k5,upl
dump dump-main-b4 trashcan-15e5457 4 dec-ref14.ids
dump dump-upl-b4 trashcan-dk2-v4 4 dec-ref14.ids TRASHCAN_Q4_VAR=upl
echo "dk2-s6 done"
