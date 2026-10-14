#!/bin/bash
# Stage-bench of one Mac's two stages (layers 0-3 with the n-gram module on GPU 0, 4-7 on GPU 1; each holds one QSA
# layer) at long positions: the 128-token chunk that ends a prefill of START tokens and 16 decode steps after it,
# with per-dispatch GPU timestamps. EXTRA='K=V ...' adds environment variables. Run under a gpuq lease in
# ~/trashcan-exp/flashnext.
# usage: [EXTRA='K=V'] long-sb.sh BINARY TAG START...
cd ~/trashcan-exp/flashnext || exit 1
bin=$1 tag=$2
shift 2
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
mkdir -p results/clocks
for s in "$@"; do
    name=long-sb-$tag-$s
    scripts/with-clocks.sh results/clocks/$name.txt sudo -n env TRASHCAN_Q4_PROFILE=1 ${EXTRA:-} \
        python3 scripts/latency-run.py ./$bin qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context 32768 \
        --batch 128 --prefill ${PREFILL:-1} --decode 16 --start $s > results/$name.json 2> results/$name.err
    echo "$name rc=$? $(tail -1 results/$name.err)"
done
