#!/bin/bash
# dk2 session 10 (one Mac, GPU 0, under a gpuq lease): c2 = c1 on 9baedca9 + light barriers in decode-sized graphs
# (shader write -> none; uniform reads of activations through vector loads). barrier-bench (cache probe with 16 KB and
# 1 KB); decode of layers 4-7 after 512 tokens: main (9baedca), c1 (trashcan-dk2-bdiag, default barriers) and c2
# alternated twice, then profiled, also after 4096 tokens and in chunks of 4; dumps of main and c2 (14 tokens one at
# a time and in chunks of 4, 100 tokens in chunks of 16, 2101 tokens one at a time).
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
R=results/dk2/s10
mkdir -p $R
for c in 0 1; do
    f=/sys/class/drm/card$c/device/power_dpm_force_performance_level
    if echo high | sudo -n tee $f > /dev/null; then echo "card$c high ok"; else echo "card$c high FAILED"; fi
    echo auto | sudo -n tee $f > /dev/null
done
SB="qwen4 stage-bench $M --ranges 4:8 --gpus 0 --context 32768 --decode 64"
run() {
    name=$1 bin=$2 args=$3
    shift 3
    env "$@" scripts/with-clocks.sh $R/$name.clk ./$bin $SB $args > $R/$name.json 2> $R/$name.err
    echo "$name rc=$? $(grep clocks $R/$name.err | cut -c1-160)"
}
dump() {
    name=$1 bin=$2 b=$3 ids=$4
    rm -rf $R/$name
    ./$bin qwen4 stage-bench $M --ranges 4:8 --gpus 0 --context 32768 --batch $b --tokens-file $ids \
        --out $R/$name > $R/$name.json 2> $R/$name.err
    echo "$name rc=$?"
}
scripts/with-clocks.sh $R/barrier.clk ./trashcan-dk2-c2 qwen4 barrier-bench --gpu 0 --iterations 400 \
    > $R/barrier.json 2> $R/barrier.err
echo "barrier rc=$?"
D="--start 512 --batch 128 --prefill 0"
for r in 1 2; do
    run main-$r trashcan-9baedca "$D"
    run c1-$r trashcan-dk2-bdiag "$D"
    run c2-$r trashcan-dk2-c2 "$D"
done
for b in main:trashcan-9baedca c1:trashcan-dk2-bdiag c2:trashcan-dk2-c2; do
    run ${b%%:*}-p ${b#*:} "$D" TRASHCAN_Q4_PROFILE=1
done
L="--start 4096 --batch 128 --prefill 0"
run c1-4096p trashcan-dk2-bdiag "$L" TRASHCAN_Q4_PROFILE=1
run c2-4096p trashcan-dk2-c2 "$L" TRASHCAN_Q4_PROFILE=1
V="--start 512 --batch 4 --prefill 8"
run c1-b4 trashcan-dk2-bdiag "$V" TRASHCAN_Q4_PROFILE=1
run c2-b4 trashcan-dk2-c2 "$V" TRASHCAN_Q4_PROFILE=1
echo "dk2-s10 timing done"
dump dump-main-b1 trashcan-9baedca 1 dec-ref14.ids
dump dump-c2-b1 trashcan-dk2-c2 1 dec-ref14.ids
dump dump-main-b4 trashcan-9baedca 4 dec-ref14.ids
dump dump-c2-b4 trashcan-dk2-c2 4 dec-ref14.ids
dump dump-main-p16 trashcan-9baedca 16 dec-p100.ids
dump dump-c2-p16 trashcan-dk2-c2 16 dec-p100.ids
dump dump-main-l2101 trashcan-9baedca 1 long-2101.ids
dump dump-c2-l2101 trashcan-dk2-c2 1 long-2101.ids
echo "dk2-s10 done"
