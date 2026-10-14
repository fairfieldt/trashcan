#!/bin/bash
# dk2 session 4 (one Mac, GPU 0, under a gpuq lease, from ~/trashcan-exp/flashnext): v3 (gate/up with 2 + 2 rows per
# wave for one token) and v3 with the fused one-token mixer (TRASHCAN_Q4_VAR=mixf) against main, decode of layers
# 4-7 after 512 tokens alternated, profiles; stage dumps of the 14-token prompt one token at a time against main's.
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
R=results/dk2/s4
mkdir -p $R
for c in 0 1; do
    f=/sys/class/drm/card$c/device/power_dpm_force_performance_level
    if echo high | sudo -n tee $f > /dev/null; then echo "card$c high ok"; else echo "card$c high FAILED"; fi
    echo auto | sudo -n tee $f > /dev/null
done
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
run main-1 trashcan-15e5457
run v3-1 trashcan-dk2-v3
run v3f-1 trashcan-dk2-v3 TRASHCAN_Q4_VAR=mixf
run main-2 trashcan-15e5457
run v3f-2 trashcan-dk2-v3 TRASHCAN_Q4_VAR=mixf
run v3-2 trashcan-dk2-v3
run main-p trashcan-15e5457 TRASHCAN_Q4_PROFILE=1
run v3-p trashcan-dk2-v3 TRASHCAN_Q4_PROFILE=1
run v3f-p trashcan-dk2-v3 TRASHCAN_Q4_PROFILE=1 TRASHCAN_Q4_VAR=mixf
dump dump-main-b1 trashcan-15e5457 1 dec-ref14.ids
dump dump-v3-b1 trashcan-dk2-v3 1 dec-ref14.ids
dump dump-v3f-b1 trashcan-dk2-v3 1 dec-ref14.ids TRASHCAN_Q4_VAR=mixf
dump dump-main-p16 trashcan-15e5457 16 dec-p100.ids
dump dump-v3f-p16 trashcan-dk2-v3 16 dec-p100.ids TRASHCAN_Q4_VAR=mixf
echo "dk2-s4 done"
