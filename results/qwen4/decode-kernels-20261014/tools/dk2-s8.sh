#!/bin/bash
# dk2 session 8 (one Mac, GPU 0, under a gpuq lease): the candidate c1 (gate/up 2 + 2 rows for one token, the QSA
# post-projection kernels in one group) against main: decode of layers 4-7 after 512 tokens (two alternated pairs,
# then profiled), after 4096 tokens (profiled pair), and 2- and 4-token chunks (verification-shaped, profiled pairs).
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
R=results/dk2/s8
mkdir -p $R
for c in 0 1; do
    f=/sys/class/drm/card$c/device/power_dpm_force_performance_level
    if echo high | sudo -n tee $f > /dev/null; then echo "card$c high ok"; else echo "card$c high FAILED"; fi
    echo auto | sudo -n tee $f > /dev/null
done
SB="qwen4 stage-bench $M --ranges 4:8 --gpus 0 --context 32768 --decode 64"
run() {
    name=$1 bin=$2 extra=$3
    shift 3
    env "$@" scripts/with-clocks.sh $R/$name.clk ./$bin $SB $extra > $R/$name.json 2> $R/$name.err
    echo "$name rc=$? $(grep clocks $R/$name.err | cut -c1-160)"
}
D="--start 512 --batch 128 --prefill 0"
run main-1 trashcan-15e5457 "$D"
run c1-1 trashcan-dk2-c1 "$D"
run main-2 trashcan-15e5457 "$D"
run c1-2 trashcan-dk2-c1 "$D"
run main-p trashcan-15e5457 "$D" TRASHCAN_Q4_PROFILE=1
run c1-p trashcan-dk2-c1 "$D" TRASHCAN_Q4_PROFILE=1
L="--start 4096 --batch 128 --prefill 0"
run main-4096p trashcan-15e5457 "$L" TRASHCAN_Q4_PROFILE=1
run c1-4096p trashcan-dk2-c1 "$L" TRASHCAN_Q4_PROFILE=1
for b in 2 4; do
    V="--start 512 --batch $b --prefill 8"
    run main-b$b trashcan-15e5457 "$V" TRASHCAN_Q4_PROFILE=1
    run c1-b$b trashcan-dk2-c1 "$V" TRASHCAN_Q4_PROFILE=1
done
echo "dk2-s8 done"
# Q8 K/V cache (the sparse path's other variant): dumps of main and c1.
for b in 1 16; do
    ids=dec-ref14.ids; [ $b = 16 ] && ids=dec-p100.ids
    for bin in trashcan-15e5457 trashcan-dk2-c1; do
        name=dump-q8-$bin-b$b
        rm -rf $R/$name
        ./$bin qwen4 stage-bench $M --ranges 4:8 --gpus 0 --context 32768 --kv q8 --batch $b --tokens-file $ids \
            --out $R/$name > $R/$name.json 2> $R/$name.err
        echo "$name rc=$?"
    done
done
echo "dk2-s8 q8 done"
