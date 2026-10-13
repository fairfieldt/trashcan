#!/bin/bash
# Run on a Mac under a gpuq lease, from ~/trashcan-exp/flashnext: the Q8_0 GEMM microbenchmark of trashcan-pf2 for
# each token count, kernels compared bit for bit with the first; then the shader statistics and disassembly.
# usage: pf2-gemm.sh TAG "N1 N2 ..." SHAPES KERNELS [ITERATIONS]
set -uo pipefail
cd ~/trashcan-exp/flashnext
tag=$1 ns=$2 shapes=$3 kernels=$4 it=${5:-20}
mkdir -p results/clocks
out=results/pf2-gemm-$tag.jsonl
: > $out
for n in $ns; do
    scripts/with-clocks.sh results/clocks/pf2-gemm-$tag-n$n.txt ./${BIN:-trashcan-pf2} qwen-gemm-bench --shapes $shapes --n $n \
        --kernels $kernels --iterations $it --gpu 0 >> $out 2> results/pf2-gemm-$tag-n$n.err
    echo "n=$n rc=$? $(tail -1 results/pf2-gemm-$tag-n$n.err)"
done
MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaders,shaderstats,nocache ./${BIN:-trashcan-pf2} qwen-gemm-bench --shapes 2560x2560 --n 16 --kernels qwen4_q8_gemm \
    --iterations 1 --gpu 0 > /dev/null 2> results/pf2-gemm-$tag-isa.txt
echo "isa rc=$?"
