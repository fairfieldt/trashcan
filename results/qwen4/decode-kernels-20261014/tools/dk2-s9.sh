#!/bin/bash
# dk2 session 9 (one Mac, GPU 0, under a gpuq lease): which cache a decode barrier must keep. c1 on 9baedca9 with the
# barrier after each decode dispatch chosen by TRASHCAN_Q4_BARRIER (temporary diagnostic): 0 main's (shader write ->
# shader read and write), 1 execution only, 2 none -> uniform read (L1 and K$), 3 none -> indirect-command read (K$
# only), 4 shader write -> none (RADV GFX6: L2 write-back and invalidation with the L1). Decode of layers 4-7 after
# 512 tokens, two alternated rounds and one profiled; main (trashcan-9baedca) once; barrier-bench.
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
R=results/dk2/s9
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
scripts/with-clocks.sh $R/barrier.clk ./trashcan-dk2-bdiag qwen4 barrier-bench --gpu 0 --iterations 400 \
    > $R/barrier.json 2> $R/barrier.err
echo "barrier rc=$?"
run main-1 trashcan-9baedca
for r in 1 2; do
    for m in 0 1 2 3 4; do run m$m-$r trashcan-dk2-bdiag TRASHCAN_Q4_BARRIER=$m; done
done
for m in 0 1 2 3 4; do run m$m-p trashcan-dk2-bdiag TRASHCAN_Q4_BARRIER=$m TRASHCAN_Q4_PROFILE=1; done
echo "dk2-s9 done"
