#!/bin/bash
# Measure n-gram table placement on incite: a resident Q4_0 copy in RAM, and rows read from the Q8_0 GGUF shard on
# the SSD (O_DIRECT, and through a page cache that starts cold), for corpus token ids and for uniformly random
# tokens. Disk-heavy: run the whole script under one gpuq lease. A resident Q8_0 copy (50.7 GiB) is not tried:
# with 62 GiB of RAM it risks the OOM killer.
# usage (on incite, from ~/trashcan-exp/qwen-flash): gpuq run ID -- scripts/qwen4-ngram-bench.sh OUTDIR
set -u
out=${1:-results/ngram}
bin=./bin/trashcan
meta=$HOME/models/qwen3.8-flash-next/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf
table=$HOME/models/qwen3.8-flash-next/Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf
ids=$HOME/trashcan-exp/qwen-flash/corpus.ids
mkdir -p "$HOME/trashcan-exp/qwen-flash/$out"
for mode in direct buffered-cold ram-q4; do
    for src in corpus random; do
        arg=(); [ $src = corpus ] && arg=(--ids "$ids")
        $bin qwen4 ngram-bench "$meta" "$table" --mode $mode \
            --threads 16 --tokens 8192 --batch 512 "${arg[@]}" > "$HOME/trashcan-exp/qwen-flash/$out/$mode-$src.json" \
            2> "$HOME/trashcan-exp/qwen-flash/$out/$mode-$src.log"
        cat "$HOME/trashcan-exp/qwen-flash/$out/$mode-$src.json"
    done
done
