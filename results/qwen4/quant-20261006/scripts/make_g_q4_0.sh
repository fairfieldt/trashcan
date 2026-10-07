#!/bin/bash
# The GPU validation GGUF: every matrix the qwen4 GPU stage reads in Q4_0 (experts, mixers, DeltaNet, attention,
# shared expert, n-gram projections), the head in Q6_K; norms, routers and the n-gram table as in the source.
cd /Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash
# shellcheck disable=SC2046
THREADS=${THREADS:-8} ./quant.sh G-q4_0 $(cat ne-q4_0.args) \
    --tensor-type ffn_gate_exps=q4_0 --tensor-type ffn_up_exps=q4_0 --tensor-type ffn_down_exps=q4_0 \
    --output-tensor-type q6_k
