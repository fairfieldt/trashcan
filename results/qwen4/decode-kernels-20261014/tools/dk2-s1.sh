#!/bin/bash
# dk2 session 1 (one Mac, GPU 0, under a gpuq lease, from ~/trashcan-exp/flashnext): the barrier microbenchmark, then
# decode profiles of layers 4-7 (DeltaNet 4-6, QSA 7) after 512 tokens: main (trashcan-15e5457) without and with
# per-dispatch timestamps, and the diagnostic binary (main + TRASHCAN_Q4_DIAG) with every dispatch recorded twice
# and/or execution-only (exec) or L1-only (lite) barriers; stage dumps of the 14-token prompt one token at a time
# with main and with lite barriers.
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
R=results/dk2/s1
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
    name=$1 bin=$2
    shift 2
    rm -rf $R/$name
    env "$@" ./$bin qwen4 stage-bench $M --ranges 4:8 --gpus 0 --context 32768 --batch 1 --tokens-file dec-ref14.ids \
        --out $R/$name > $R/$name.json 2> $R/$name.err
    echo "$name rc=$?"
}
scripts/with-clocks.sh $R/barrier.clk ./trashcan-dk2-diag qwen4 barrier-bench --gpu 0 --iterations 400 \
    > $R/barrier.json 2> $R/barrier.err
echo "barrier rc=$? $(grep clocks $R/barrier.err | cut -c1-160)"
run main-np trashcan-15e5457
run main-p trashcan-15e5457 TRASHCAN_Q4_PROFILE=1
run diag-p trashcan-dk2-diag TRASHCAN_Q4_PROFILE=1
run diag-twice-p trashcan-dk2-diag TRASHCAN_Q4_PROFILE=1 TRASHCAN_Q4_DIAG=twice
run diag-exec-p trashcan-dk2-diag TRASHCAN_Q4_PROFILE=1 TRASHCAN_Q4_DIAG=exec
run diag-lite-p trashcan-dk2-diag TRASHCAN_Q4_PROFILE=1 TRASHCAN_Q4_DIAG=lite
run diag-twlite-p trashcan-dk2-diag TRASHCAN_Q4_PROFILE=1 TRASHCAN_Q4_DIAG=twice,lite
run diag-lite-np trashcan-dk2-diag TRASHCAN_Q4_DIAG=lite
run diag-exec-np trashcan-dk2-diag TRASHCAN_Q4_DIAG=exec
run main-np2 trashcan-15e5457
dump dump-main trashcan-15e5457
dump dump-lite trashcan-dk2-diag TRASHCAN_Q4_DIAG=lite
echo "dk2-s1 done"
