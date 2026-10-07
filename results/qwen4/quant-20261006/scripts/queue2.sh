#!/bin/bash
# Dense-weight sensitivity: one group of dense matrices at Q4_0 at a time (everything else Q8_0), after queue 1.
cd /Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash || exit 1
while pgrep -f "kld_queue.sh E-q4_0" > /dev/null; do sleep 30; done
exec ./kld_queue.sh \
  A-hc "--tensor-type hc_attn_down=q4_0 --tensor-type hc_attn_up=q4_0 --tensor-type hc_attn_inject=q4_0 --tensor-type hc_ffn_down=q4_0 --tensor-type hc_ffn_up=q4_0 --tensor-type hc_ffn_inject=q4_0 --tensor-type output_hc_down=q4_0 --tensor-type output_hc_up=q4_0" \
  A-gdn "--tensor-type attn_qkv=q4_0 --tensor-type attn_gate=q4_0 --tensor-type ssm_out=q4_0" \
  A-attn "--tensor-type attn_q.weight=q4_0 --tensor-type attn_k.weight=q4_0 --tensor-type attn_v.weight=q4_0 --tensor-type attn_output=q4_0" \
  A-shexp "--tensor-type ffn_gate_shexp=q4_0 --tensor-type ffn_up_shexp=q4_0 --tensor-type ffn_down_shexp=q4_0"
