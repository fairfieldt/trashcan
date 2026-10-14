#!/bin/bash
# dk2 session 5 (one Mac, GPU 0, under a gpuq lease): where the fused one-token mixer's time goes: normal, without the
# wait for the flag (wrong results), and with the second half exiting after its loads (timing only).
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
R=results/dk2/s5
mkdir -p $R
SB="qwen4 stage-bench $M --ranges 4:8 --gpus 0 --context 32768 --start 512 --decode 64 --batch 128 --prefill 0"
run() {
    name=$1 bin=$2
    shift 2
    env "$@" scripts/with-clocks.sh $R/$name.clk ./$bin $SB > $R/$name.json 2> $R/$name.err
    echo "$name rc=$? $(grep clocks $R/$name.err | cut -c1-160)"
}
run f-p trashcan-dk2-v3d TRASHCAN_Q4_PROFILE=1 TRASHCAN_Q4_VAR=mixf
run fns-p trashcan-dk2-v3d TRASHCAN_Q4_PROFILE=1 TRASHCAN_Q4_VAR=mixf,nospin
run flo-p trashcan-dk2-v3d TRASHCAN_Q4_PROFILE=1 TRASHCAN_Q4_VAR=mixf,loadonly
echo "dk2-s5 done"
