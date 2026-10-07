#!/bin/bash
# Six-Mac candidate: IQ4_XS gate/up and IQ4_NL down for routed and shared experts (the GPU stacks the shared expert
# with the routed ones), dense matrices as in the Q8_0 source, Q6_K head.
cd /Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash || exit 1
THREADS=${THREADS:-12} ./quant.sh G-iq4 \
    --tensor-type ffn_gate_exps=iq4_xs --tensor-type ffn_up_exps=iq4_xs --tensor-type ffn_down_exps=iq4_nl \
    --tensor-type ffn_gate_shexp=iq4_xs --tensor-type ffn_up_shexp=iq4_xs --tensor-type ffn_down_shexp=iq4_nl \
    --output-tensor-type q6_k
