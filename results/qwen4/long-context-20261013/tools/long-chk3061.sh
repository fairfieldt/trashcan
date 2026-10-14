#!/bin/bash
# The 3061-token val prompt (long.ids) through invite's layer ranges (32:36 on GPU 0, 36:40 on GPU 1; QSA layers 35 and
# 39) on one Mac, dumps of the last chunk's last token, sustained clocks logged:
#  a     r21, chunks of 64 (--batch 64), one chunk at a time;
#  a2    r21q, chunks of 64 with the chain's capacity of 128 (TRASHCAN_Q4_DUMP_CHUNK=64), one chunk at a time;
#  b     r21q, the same with every chunk but the last queued through Local::push (two in flight per GPU, as on a node);
#  main  trashcan-5e48d7e, as a.
# Run under a gpuq lease in ~/trashcan-exp/flashnext. usage: long-chk3061.sh TAG
cd ~/trashcan-exp/flashnext || exit 1
tag=$1
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
R="--ranges 32:36,36:40 --gpus 0,1 --context 32768 --tokens-file long.ids"
mkdir -p results/clocks
one() { # name binary batch env...
    local name=long-chk-$tag-$1 bin=$2 batch=$3
    shift 3
    rm -rf "results/$name"
    scripts/with-clocks.sh results/clocks/$name.txt env "$@" ./$bin qwen4 stage-bench $M $R --batch $batch \
        --out "results/$name" > "results/$name.log" 2>&1
    echo "$name rc=$? $(grep -h 'stage_dump\|rror' "results/$name.log" | head -2 | tr '\n' ' ')$(tail -1 "results/$name.log" | cut -c1-120)"
}
one a trashcan-long-r21 64 X=1
one a2 trashcan-long-r21q 128 TRASHCAN_Q4_DUMP_CHUNK=64
one b trashcan-long-r21q 128 TRASHCAN_Q4_DUMP_CHUNK=64 TRASHCAN_Q4_DUMP_QUEUED=1
one main trashcan-5e48d7e 64 X=1
