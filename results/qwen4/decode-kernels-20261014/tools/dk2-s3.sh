#!/bin/bash
# dk2 session 3 (one Mac, GPU 0, under a gpuq lease, from ~/trashcan-exp/flashnext): which caches each barrier
# variant invalidates (barrier-bench cache probe); v2 kernels (mixer halves, down ids in LDS, decode attention V rows
# four at a time) against main, decode of layers 4-7 after 512 tokens alternated, profiles at 1 and 4 tokens;
# temporary variants (TRASHCAN_Q4_VAR): layout probes for gate/up and down (timing only), out projections with
# 4 rows per wave, gate/up with 2 + 2 rows per wave; stage dumps against main's.
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
R=results/dk2/s3
mkdir -p $R
for c in 0 1; do
    f=/sys/class/drm/card$c/device/power_dpm_force_performance_level
    if echo high | sudo -n tee $f > /dev/null; then echo "card$c high ok"; else echo "card$c high FAILED"; fi
    echo auto | sudo -n tee $f > /dev/null
done
SB="qwen4 stage-bench $M --ranges 4:8 --gpus 0 --context 32768 --start 512 --decode 64"
run() {
    name=$1 bin=$2 extra=$3
    shift 3
    env "$@" scripts/with-clocks.sh $R/$name.clk ./$bin $SB $extra > $R/$name.json 2> $R/$name.err
    echo "$name rc=$? $(grep clocks $R/$name.err | cut -c1-160)"
}
dump() {
    name=$1 bin=$2 b=$3 ids=$4
    rm -rf $R/$name
    ./$bin qwen4 stage-bench $M --ranges 4:8 --gpus 0 --context 32768 --batch $b --tokens-file $ids \
        --out $R/$name > $R/$name.json 2> $R/$name.err
    echo "$name rc=$?"
}
scripts/with-clocks.sh $R/barrier.clk ./trashcan-dk2-v2 qwen4 barrier-bench --gpu 0 --iterations 400 \
    > $R/barrier.json 2> $R/barrier.err
echo "barrier rc=$? $(grep clocks $R/barrier.err | cut -c1-160)"
D="--batch 128 --prefill 0"
run main-1 trashcan-15e5457 "$D"
run v2-1 trashcan-dk2-v2 "$D"
run main-2 trashcan-15e5457 "$D"
run v2-2 trashcan-dk2-v2 "$D"
run main-p trashcan-15e5457 "$D" TRASHCAN_Q4_PROFILE=1
run v2-p trashcan-dk2-v2 "$D" TRASHCAN_Q4_PROFILE=1
run v2-lp-p trashcan-dk2-v2 "$D" TRASHCAN_Q4_PROFILE=1 TRASHCAN_Q4_VAR=gu_lp,down_lp,out4
run v2-gu2-p trashcan-dk2-v2 "$D" TRASHCAN_Q4_PROFILE=1 TRASHCAN_Q4_VAR=gu2
# 4-token chunks at position 0 (verification-shaped), profiled.
V="--batch 4 --prefill 8"
run main-b4 trashcan-15e5457 "$V" TRASHCAN_Q4_PROFILE=1
run v2-b4 trashcan-dk2-v2 "$V" TRASHCAN_Q4_PROFILE=1
dump dump-main-b1 trashcan-15e5457 1 dec-ref14.ids
dump dump-v2-b1 trashcan-dk2-v2 1 dec-ref14.ids
dump dump-main-b4 trashcan-15e5457 4 dec-ref14.ids
dump dump-v2-b4 trashcan-dk2-v2 4 dec-ref14.ids
dump dump-v2-p16 trashcan-dk2-v2 16 dec-p100.ids
echo "dk2-s3 done"
