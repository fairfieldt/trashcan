#!/bin/bash
# Equivalence of two binaries' default paths on one Mac (G-iq4, 32K context, layers 0-3 on GPU 0 and 4-7 on GPU 1):
# dumps of the last chunk's last token for each prompt long-N.ids, under results/long-equiv-TAG-{a,b}-N.
# KV=q8 (or k16v8) runs with that K/V cache. Run under a gpuq lease in ~/trashcan-exp/flashnext.
# usage: [KV=q8] long-equiv.sh BIN_A BIN_B TAG N...
cd ~/trashcan-exp/flashnext || exit 1
a=$1 b=$2 tag=$3
shift 3
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
for n in "$@"; do
    for side in a b; do
        bin=$a; [ $side = b ] && bin=$b
        out=results/long-equiv-$tag-$side-$n
        rm -rf "$out"
        ./$bin qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch 128 --kv ${KV:-f16} \
            --tokens-file long-$n.ids --out "$out" > "$out.log" 2>&1
        echo "$n $side rc=$?"
    done
done
