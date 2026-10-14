#!/bin/bash
# dk2 session 11 (one Mac, GPU 0, under a gpuq lease): c3 = c2 + qwen4_moe_down_p for one token (5 rows per wave, the
# next step's weights loaded during this step's products) against c2: decode of layers 4-7 after 512 tokens,
# alternated twice, then profiled; dumps of c3 (14 tokens and 2101 tokens one at a time) against main's from s10.
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
R=results/dk2/s11
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
D="--start 512 --batch 128 --prefill 0"
for r in 1 2; do
    run c2-$r trashcan-dk2-c2 "$D"
    run c3-$r trashcan-dk2-c3 "$D"
done
run c2-p trashcan-dk2-c2 "$D" TRASHCAN_Q4_PROFILE=1
run c3-p trashcan-dk2-c3 "$D" TRASHCAN_Q4_PROFILE=1
dump dump-c3-b1 trashcan-dk2-c3 1 dec-ref14.ids
dump dump-c3-l2101 trashcan-dk2-c3 1 long-2101.ids
echo "dk2-s11 done"
