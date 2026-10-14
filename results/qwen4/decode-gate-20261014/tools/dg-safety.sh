#!/bin/bash
# Decode gate, one-Mac safety (gpuq run ID -- bash THIS, from ~/trashcan-exp/flashnext), on the model's first 6 layers
# (TRASHCAN_Q4_LAYERS=6) with a node and the driver on one GPU each (localhost), both ways round:
# 1. gates that give up (after 0.1 s): the driver pauses 400 ms after the notice of the forward at 120 (the node's gate
#    gives up and its stage runs again without a gate), then 400 ms after submitting its head scoring ahead at 140 (its
#    own head gate gives up); the 64 greedy tokens must match a run without pauses;
# 2. kill -9 with gated graphs on the GPU: the node 0.05 s into a 3 s pause after the notice (its stage waits behind
#    its gate), the driver 0.05 s into a 3 s pause after submitting its head scoring ahead, then the node and the
#    driver at a random moment of a long decode; after each, both GPUs must still run work (`gate-skip-check`).
# Never two engine processes on one GPU: each starts only on a GPU of its own with the VRAM it loads free. The kernel
# log (amdgpu faults, PCIe AER) is checked after every step; the first match stops everything.
cd ~/trashcan-exp/flashnext || exit 1
. ./dg-common.sh
echo "kernel $(uname -r)"
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
B=${B:-./trashcan-dg7}
export TRASHCAN_Q4_LAYERS=6
C="--context 4096 --batch 128 --head-split 2 --cuts 3"
PORT=7181
trap 'kill -9 $(jobs -p) 2>/dev/null; wait 2>/dev/null' EXIT
# start_node NAME GPU: a node on GPU (in the background; its pid in $np).
start_node() {
    need_vram "$2" $NEED
    $B qwen4 node $M --listen 127.0.0.1:$PORT --gpus "$2" > "$R/$1-node.log" 2>&1 &
    np=$!
}
# pair NAME NODE_GPU DRIVER_GPU STALL ARGS...: a node, then the driver with TRASHCAN_Q4_STALL=STALL.
pair() {
    local name=$1 ng=$2 dg=$3 stall=$4 np
    shift 4
    [ "$ng" != "$dg" ] || { echo "STOP: $name would put the node and the driver on GPU $ng"; exit 5; }
    if [ -n "${DRY:-}" ]; then
        echo "DRY $name: node --gpus $ng; driver TRASHCAN_Q4_STALL=$stall qwen4 $* --gpus $dg --nodes 127.0.0.1:$PORT $C"
        return
    fi
    start_node "$name" "$ng"
    need_vram "$dg" $NEED
    TRASHCAN_Q4_STALL=$stall step "$name" $B qwen4 "$@" --gpus "$dg" --nodes 127.0.0.1:$PORT $C
    kill $np 2>/dev/null
    wait $np 2>/dev/null
    check "$name-node"
    sweeps "$R/$name.err" "$R/$name-node.log"
}
# killtest NAME NODE_GPU DRIVER_GPU STALL WHO DELAY: start a node and the driver (a long greedy decode), then kill -9 WHO
# (node or driver) DELAY s after the driver logs its stall (or, with STALL "-", DELAY s after it starts); then the
# other process, and a check that both GPUs still run work.
killtest() {
    local name=$1 ng=$2 dg=$3 stall=$4 who=$5 delay=$6 np dp t0 k
    [ "$ng" != "$dg" ] || { echo "STOP: $name would put the node and the driver on GPU $ng"; exit 5; }
    if [ -n "${DRY:-}" ]; then
        echo "DRY $name: node --gpus $ng, driver --gpus $dg (stall $stall), kill -9 the $who after $delay s"
        return
    fi
    start_node "$name" "$ng"
    need_vram "$dg" $NEED
    TRASHCAN_Q4_STALL=$stall $B qwen4 validate $M "$R/$name" --greedy 3000 --tokens-file p100.ids --gpus "$dg" \
        --nodes 127.0.0.1:$PORT $C > "$R/$name.out" 2> "$R/$name.err" &
    dp=$!
    t0=$(date +%s.%N)
    echo "$name: node on GPU $ng (pid $np), driver on GPU $dg (pid $dp), started $(date +%T.%N | cut -c1-12)"
    if [ "$stall" != - ]; then
        until grep -q "stall at" "$R/$name.err"; do
            sleep 0.01
            kill -0 $dp 2>/dev/null || { echo "$name: the driver ended before its stall"; break; }
        done
    fi
    sleep "$delay"
    if [ "$who" = node ]; then kill -9 $np; else kill -9 $dp; fi
    echo "$name: kill -9 the $who at $(date +%T.%N | cut -c1-12), \
$(awk -v a="$(date +%s.%N)" -v b="$t0" 'BEGIN { printf "%.2f", a - b }') s after the start"
    sleep 3
    kill -9 $np $dp 2>/dev/null
    wait $np $dp 2>/dev/null
    check "$name"
    sweeps "$R/$name.err" "$R/$name-node.log"
    for k in 0 1; do
        need_vram "$k" 64
        step "$name-after-gpu$k" $B qwen4 gate-skip-check --gpu $k --dispatches 10 --reps 20
    done
}
power
for g in 0 1; do
    pair "plain-$g" $((1 - g)) $g none validate $M "$R/plain-$g" --greedy 64 --tokens-file p100.ids
    pair "stall-fwd-$g" $((1 - g)) $g 120:400 validate $M "$R/stall-fwd-$g" --greedy 64 --tokens-file p100.ids
    pair "stall-head-$g" $((1 - g)) $g 140:400:heads validate $M "$R/stall-head-$g" --greedy 64 --tokens-file p100.ids
done
for g in 0 1; do
    killtest "kill-node-$g" $((1 - g)) $g 120:3000 node 0.05
    killtest "kill-driver-$g" $((1 - g)) $g 120:3000:heads driver 0.05
    killtest "kill-random-node-$g" $((1 - g)) $g - node "$((RANDOM % 4 + 8)).$((RANDOM % 10))"
    killtest "kill-random-driver-$g" $((1 - g)) $g - driver "$((RANDOM % 4 + 8)).$((RANDOM % 10))"
done
leftovers
# Killed processes leave their clock level behind: back to auto.
power
echo "safety done; kernel log matches $(hits | wc -l) (baseline ${base:-})"
