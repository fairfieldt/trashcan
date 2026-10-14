#!/bin/bash
# Kernel check on one Mac (G-iq4, 32K context, layers 0-3 on GPU 0 and 4-7 on GPU 1, so layers 4-7 also see GPU 0's
# output copy): dumps of the last chunk, every token's rows (TRASHCAN_Q4_DUMP_ALL=1), with the kernels before this
# round (TRASHCAN_Q4_QSA=old TRASHCAN_Q4_COPY_OLD=1) and after (new, with EXTRA='K=V ...' added), for each prompt
# long-N.ids. KV=q8 (or k16v8) runs with that K/V cache. Run under a gpuq lease in ~/trashcan-exp/flashnext.
# usage: [EXTRA='K=V'] [KV=q8] long-unit.sh BINARY TAG N...
cd ~/trashcan-exp/flashnext || exit 1
bin=$1 tag=$2
shift 2
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
for n in "$@"; do
    for mode in old new; do
        out=results/long-dump-$tag-$n-$mode
        rm -rf "$out"
        if [ $mode = old ]; then env="TRASHCAN_Q4_QSA=old TRASHCAN_Q4_COPY_OLD=1"; else env=${EXTRA:-}; fi
        env TRASHCAN_Q4_DUMP_ALL=1 $env ./$bin qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context 32768 \
            --batch 128 --kv ${KV:-f16} --tokens-file long-$n.ids --out "$out" > "$out.log" 2>&1
        echo "$n $mode rc=$?"
    done
done
