#!/bin/bash
# One stage-bench load timing on this Mac: layers 0-7 on both GPUs, as in the load-time task.
# usage: load-sb.sh NAME BINARY warm|cold [EXTRA STAGE-BENCH ARGS...]   (run under gpuq run; cold drops the page cache)
cd ~/trashcan-exp/flashnext || exit 1
name=$1 bin=$2 mode=$3
shift 3
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
if [ "$mode" = cold ]; then
    sync
    echo 1 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null
fi
s=$(date +%s.%N)
./$bin qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch 128 \
    --head-rows 0:21504,21504:43008 --decode 1 --start 16 --prefill 0 "$@" > results/load-$name.json 2> results/load-$name.err
rc=$?
e=$(date +%s.%N)
echo "$name $bin $mode rc $rc wall $(echo "$e - $s" | bc) s"
grep LOADPROF results/load-$name.err
