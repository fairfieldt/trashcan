#!/bin/bash
# Fan ramp under GPU work (2026-10-27): on MAC (default invite, the Mac whose GPU 0 runs hottest), from idle at the
# curve's idle speed, sample fan and temperatures every second while three Flash-Next stage prefill runs (layers 0-8 on
# both GPUs, 200 chunks of 128 tokens each) go under a gpuq lease, then for HOLD+60 s after, to see the step to full
# speed, the 120 s hold and the step-down.
W0=/Users/fairfieldt/dev/claude-work/flashnext-100k-20261027
MAC=${MAC:-invite}
OUT=$W0/fan/ramp-$MAC.log
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
id=$(gpuq --on $MAC request "fan ramp test" --est 9 --owner "fan curve" | sed -n 's/^Ticket #\([0-9]*\) created.*/\1/p')
[ -n "$id" ] || { echo "lease request failed"; exit 1; }
trap 'gpuq --on $MAC release $id -r "fan ramp: stopped" > /dev/null 2>&1' EXIT
gpuq --on $MAC wait $id || exit 1
: > $OUT
$W0/fan-watch.sh $OUT 8 1 $MAC &
watch=$!
sleep 20
echo "$(date +%T) $MAC work starts" >> $OUT
ssh $MAC "gpuq run $id -- bash -c 'cd ~/trashcan-exp/flashnext; for i in 1 2 3; do ./trashcan-v5 qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch 128 --prefill 200 --decode 0 > /dev/null 2>&1 || echo stage-bench rc \$?; done'" >> $OUT 2>&1
echo "$(date +%T) $MAC work ends" >> $OUT
gpuq --on $MAC release $id -r "fan ramp: work done, watching the step-down" > /dev/null 2>&1
trap - EXIT
wait $watch
echo "ramp test done: $OUT"
