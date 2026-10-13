#!/bin/bash
# Run on one Mac under a gpuq lease, from ~/trashcan-exp/flashnext: stage-bench runs (layers 0-7 on both D700s) with
# per-dispatch profiles, and stage dumps of a prompt.
#   moe-sb.sh prof NAME BIN BATCH [ENV...]          per-dispatch profile; the BATCH-token chunk at 0 and decode at 64
#   moe-sb.sh dump NAME BIN BATCH IDS [ENV...]      stage dump of IDS in chunks of BATCH into results/moe/NAME
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
SB="qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context 32768 --head-rows 0:21504,21504:43008"
mkdir -p results/clocks results/moe
mode=$1 name=$2 bin=$3 b=$4; shift 4
if [ "$mode" = prof ]; then
    env TRASHCAN_Q4_PROFILE=1 "$@" scripts/with-clocks.sh results/clocks/$name.txt ./$bin $SB --batch $b --decode 32 \
        --start 64 --prefill 8 > results/moe/$name.json 2> results/moe/$name.err
else
    ids=$1; shift
    rm -rf results/moe/$name
    env "$@" ./$bin $SB --batch $b --tokens-file $ids --out results/moe/$name > results/moe/$name.json 2> results/moe/$name.err
fi
echo "$name rc=$? $(tail -1 results/moe/$name.err | cut -c1-200)"
