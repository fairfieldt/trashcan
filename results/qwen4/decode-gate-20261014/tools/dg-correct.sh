#!/bin/bash
# Decode gate, one-Mac correctness (gpuq run ID -- bash THIS, from ~/trashcan-exp/flashnext). The chain code on the
# model's first 6 layers (TRASHCAN_Q4_LAYERS=6: two stages of 3 layers, the final mixer and the head split over both):
# the branch (B, trashcan-dg*) against main with the same test hook (A, trashcan-dgbase*), each run once:
# - one process on both GPUs (the driver's second stage and its head scoring submitted ahead), both orders of GPUs;
# - a node on one GPU and the driver on the other, over localhost (the node's stage on the driver's notice, the
#   driver's head scoring), both ways round;
# each: validation dumps of the 100-token prompt and 2,048 greedy tokens; then MTP decoding (`mtp spec`, plain and
# with drafts, which must agree) in the two-process chain. Never two engine processes on one GPU: each starts only on
# a GPU of its own with the VRAM it loads free. The kernel log (amdgpu faults, PCIe AER) is checked after every step.
cd ~/trashcan-exp/flashnext || exit 1
. ./dg-common.sh
echo "kernel $(uname -r)"
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
A=${A:-./trashcan-dgbase3}
B=${B:-./trashcan-dg7}
export TRASHCAN_Q4_LAYERS=6
C="--context 4096 --batch 128 --head-split 2 --cuts 3"
PORT=7181
# Background nodes are stopped whatever happens.
trap 'kill $(jobs -p) 2>/dev/null; wait 2>/dev/null' EXIT
# pair NAME BIN NODE_GPU DRIVER_GPU ARGS...: a node on NODE_GPU, then the driver (`BIN qwen4 ARGS`) on DRIVER_GPU.
pair() {
    local name=$1 bin=$2 ng=$3 dg=$4 np
    shift 4
    [ "$ng" != "$dg" ] || { echo "STOP: $name would put the node and the driver on GPU $ng"; exit 5; }
    if [ -n "${DRY:-}" ]; then
        echo "DRY $name: node $bin --gpus $ng; driver $bin qwen4 $* --gpus $dg --nodes 127.0.0.1:$PORT $C"
        return
    fi
    need_vram "$ng" $NEED
    $bin qwen4 node $M --listen 127.0.0.1:$PORT --gpus "$ng" > "$R/$name-node.log" 2>&1 &
    np=$!
    need_vram "$dg" $NEED
    step "$name" $bin qwen4 "$@" --gpus "$dg" --nodes 127.0.0.1:$PORT $C
    kill $np 2>/dev/null
    wait $np 2>/dev/null
    check "$name-node"
    sweeps "$R/$name.err" "$R/$name-node.log"
}
power
for gpus in 0,1 1,0; do
    for v in A B; do
        need_vram 0 $NEED
        need_vram 1 $NEED
        step "one-$gpus-$v" ${!v} qwen4 validate $M "$R/one-$gpus-$v" --gpus $gpus $C --greedy 2048 --tokens-file p100.ids
        sweeps "$R/one-$gpus-$v.err"
    done
done
for g in 0 1; do
    for v in A B; do
        pair "two-$g-$v" ${!v} $((1 - g)) $g validate $M "$R/two-$g-$v" --greedy 2048 --tokens-file p100.ids
    done
done
for g in 0 1; do
    for v in A B; do
        pair "mtp-$g-$v" ${!v} $((1 - g)) $g mtp spec $M --mtp $MTP --draft-tokens 3 --tg 512 --runs 1 p100.ids
    done
done
leftovers
echo "correctness done; kernel log matches $(hits | wc -l) (baseline ${base:-})"
