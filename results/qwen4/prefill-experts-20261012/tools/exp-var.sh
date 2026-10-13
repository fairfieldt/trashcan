#!/bin/bash
# On a Mac under gpuq lease $1: time expert kernel variants on layers 0-3 of GPU 0 (stage-bench).
#   exp-var.sh ID BATCH "ENVSPEC ..." [STREAM]
# ENVSPEC: name:VAR=val,VAR=val (commas become separate env assignments); "main" runs trashcan-main.
set -uo pipefail
cd ~/trashcan-exp/flashnext && mkdir -p results/var
ID=$1 B=$2 specs=$3 stream=${4:-}
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
ctx=32768; [ $B -gt 128 ] && ctx=8192
for spec in $specs; do
    name=${spec%%:*}; envs=""
    [ "$spec" != "$name" ] && envs=$(echo ${spec#*:} | tr ',' ' ')
    bin=./trashcan-exp; [ $name = main ] && bin=./trashcan-main
    st=""; [ -n "$stream" ] && st="Q4X_STREAM=$stream"
    out=results/var/$name-$B${stream:+-real}
    gpuq run $ID -- env Q4X_MARK_GU=1 $st $envs $bin qwen4 stage-bench $M --ranges 0:4 --gpus 0 --context $ctx --batch $B \
        --decode 2 --start 64 --prefill 3 > $out.json 2> $out.err || { echo "$name failed"; tail -3 $out.err; continue; }
    python3 -c "
import json; d=json.load(open('$out.json')); ph={}
for s in d['prefill']:
  for l,n,ms in s['phases']: ph[n]=ph.get(n,0)+ms
print('$name B=$B ${stream:+real}'.ljust(24), 'wall', round(d['prefill_wall_ms'],1), ' '.join(f'{k} {v:.1f}' for k,v in ph.items() if k in ('router','gate_up','experts')))"
done
