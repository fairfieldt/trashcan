#!/bin/bash
# dk2 session 2 (one Mac, GPU 0, under a gpuq lease, from ~/trashcan-exp/flashnext): activation bindings coherent
# (trashcan-dk2-coh) with the engine's barriers and with execution-only barriers (TRASHCAN_Q4_DIAG=exec), against main,
# decode of layers 4-7 after 512 tokens, alternated; stage dumps (14-token prompt one token at a time and in chunks
# of 4, the 100-token prompt in chunks of 16) against main's.
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
R=results/dk2/s2
mkdir -p $R
for c in 0 1; do
    f=/sys/class/drm/card$c/device/power_dpm_force_performance_level
    if echo high | sudo -n tee $f > /dev/null; then echo "card$c high ok"; else echo "card$c high FAILED"; fi
    echo auto | sudo -n tee $f > /dev/null
done
SB="qwen4 stage-bench $M --ranges 4:8 --gpus 0 --context 32768 --batch 128 --start 512 --decode 64 --prefill 0"
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
run main-1 trashcan-15e5457
run coh-1 trashcan-dk2-coh
run cohx-1 trashcan-dk2-coh TRASHCAN_Q4_DIAG=exec
run main-2 trashcan-15e5457
run cohx-2 trashcan-dk2-coh TRASHCAN_Q4_DIAG=exec
run coh-2 trashcan-dk2-coh
run cohx-p trashcan-dk2-coh TRASHCAN_Q4_PROFILE=1 TRASHCAN_Q4_DIAG=exec
run coh-p trashcan-dk2-coh TRASHCAN_Q4_PROFILE=1
dump dump-main-b1 trashcan-15e5457 1 dec-ref14.ids
dump dump-cohx-b1 trashcan-dk2-coh 1 dec-ref14.ids TRASHCAN_Q4_DIAG=exec
dump dump-main-b4 trashcan-15e5457 4 dec-ref14.ids
dump dump-cohx-b4 trashcan-dk2-coh 4 dec-ref14.ids TRASHCAN_Q4_DIAG=exec
dump dump-main-p16 trashcan-15e5457 16 dec-p100.ids
dump dump-cohx-p16 trashcan-dk2-coh 16 dec-p100.ids TRASHCAN_Q4_DIAG=exec
echo "dk2-s2 done"
