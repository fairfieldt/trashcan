#!/bin/bash
# Draft-step profile on incite (CPU only; run under a gpuq lease so the chain is idle).
# usage: mtp2-db.sh BINARY OUTDIR [STARTS...]
cd ~/trashcan-exp/flashnext || exit 1
bin=$1 out=$2
shift 2
starts=${*:-512 2048 3072 4096}
MODEL=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
mkdir -p "$out"
for s in $starts; do
    ./$bin qwen4 mtp draft-bench $MODEL $MTP --context 8192 --start $s --steps 64 --batch 1 --threads 24 \
        --skip-head-rows 43008 > "$out/s$s.json" 2> "$out/s$s.err"
    echo "$bin start $s: p50 $(python3 -c "import json,sys; d=json.load(open('$out/s$s.json')); print(round(d['steps'][0]['p50_ms'], 3), {k: round(v, 3) for k, v in d['steps'][0]['phases_ms'].items()})")"
done
