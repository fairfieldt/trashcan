#!/bin/bash
# GPU validation GGUF with deployment-like precision: Q4_0 routed and shared experts (the GPU stacks the shared
# expert with the routed ones, so they share a format), dense matrices as in the Q8_0 source, Q6_K head.
cd /Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash || exit 1
THREADS=${THREADS:-8} ./quant.sh G-q8d \
    --tensor-type ffn_gate_exps=q4_0 --tensor-type ffn_up_exps=q4_0 --tensor-type ffn_down_exps=q4_0 \
    --tensor-type ffn_gate_shexp=q4_0 --tensor-type ffn_up_shexp=q4_0 --tensor-type ffn_down_shexp=q4_0 \
    --output-tensor-type q6_k
