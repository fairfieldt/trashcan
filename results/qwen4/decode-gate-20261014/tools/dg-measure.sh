#!/bin/bash
# Decode gate, one-Mac measurements (gpuq run ID -- bash THIS, from ~/trashcan-exp/flashnext), on the model's first 6
# layers (TRASHCAN_Q4_LAYERS=6): `bench` (100-token prompt, 512 decode steps) of the branch with its gates (on) and
# without (TRASHCAN_Q4_GATE=0, off), alternated twice, in one process on both GPUs and as a node and a driver on one GPU
# each, both ways round. Calibrated timestamps (TRASHCAN_TP_PROFILE=1) give the gates' release latency in the engine.
# Never two engine processes on one GPU: each starts only on a GPU of its own with the VRAM it loads free. The kernel
# log (amdgpu faults, PCIe AER) is checked after every step.
cd ~/trashcan-exp/flashnext || exit 1
. ./dg-common.sh
echo "kernel $(uname -r)"
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
B=${B:-./trashcan-dg7}
export TRASHCAN_Q4_LAYERS=6 TRASHCAN_TP_PROFILE=1
C="--context 4096 --batch 128 --head-split 2 --cuts 3"
PORT=7181
trap 'kill $(jobs -p) 2>/dev/null; wait 2>/dev/null' EXIT
gate() { [ "$1" = off ] && echo 0 || echo 1; }
# pair NAME GATE NODE_GPU DRIVER_GPU ARGS...
pair() {
    local name=$1 on=$2 ng=$3 dg=$4 np
    shift 4
    [ "$ng" != "$dg" ] || { echo "STOP: $name would put the node and the driver on GPU $ng"; exit 5; }
    if [ -n "${DRY:-}" ]; then
        echo "DRY $name: TRASHCAN_Q4_GATE=$(gate $on) node --gpus $ng; driver qwen4 $* --gpus $dg --nodes 127.0.0.1:$PORT $C"
        return
    fi
    need_vram "$ng" $NEED
    TRASHCAN_Q4_GATE=$(gate "$on") $B qwen4 node $M --listen 127.0.0.1:$PORT --gpus "$ng" > "$R/$name-node.log" 2>&1 &
    np=$!
    need_vram "$dg" $NEED
    TRASHCAN_Q4_GATE=$(gate "$on") step "$name" $B qwen4 "$@" --gpus "$dg" --nodes 127.0.0.1:$PORT $C
    kill $np 2>/dev/null
    wait $np 2>/dev/null
    check "$name-node"
    sweeps "$R/$name.err" "$R/$name-node.log"
}
power
for r in 1 2; do
    for g in 0 1; do
        for on in on off; do
            need_vram 0 $NEED
            need_vram 1 $NEED
            TRASHCAN_Q4_GATE=$(gate $on) step "bench-one-$g-$on-$r" $B qwen4 bench $M --gpus $g,$((1 - g)) $C --pp 100 --tg 512 \
                --tokens-file p100.ids
            sweeps "$R/bench-one-$g-$on-$r.err"
            pair "bench-two-$g-$on-$r" $on $((1 - g)) $g bench $M --pp 100 --tg 512 --tokens-file p100.ids
        done
    done
done
leftovers
echo "measure done; kernel log matches $(hits | wc -l) (baseline ${base:-})"
