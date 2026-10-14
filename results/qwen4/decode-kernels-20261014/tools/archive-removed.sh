#!/bin/bash
# Copy the removed experimental kernels of this round out of the WIP commits into results/.../removed.
set -euo pipefail
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a70283bc249c19df0
R=results/qwen4/decode-kernels-20261014/removed
mkdir -p $R
git show 37e33866:engine/shaders/qwen4_mix_down_skw.comp > $R/qwen4_mix_down_skw-prefetch.comp
git show 37e33866:engine/shaders/qwen4_mix_rows_up.comp > $R/qwen4_mix_rows_up-epilogue-first.comp
git show 37e33866:engine/shaders/qwen4_moe_down_w.comp > $R/qwen4_moe_down_w-lds-ids.comp
git show 0848dd78:engine/shaders/qwen4_attention_grouped_dec.glsl > $R/qwen4_attention_grouped_dec-v4.glsl
git show 0848dd78:engine/shaders/qwen4_moe_gu_lp.comp > $R/qwen4_moe_gu_lp.comp
git show 0848dd78:engine/shaders/qwen4_moe_down_w_lp.comp > $R/qwen4_moe_down_w_lp.comp
cp /Users/fairfieldt/dev/claude-work/flashnext-20261012/dk2/removed/qwen4_mix_fused.comp $R/
git show 02aa6e69:engine/shaders/qwen4_mix_down_k5.comp > $R/qwen4_mix_down_k5.comp
git show 02aa6e69:engine/shaders/qwen4_mix_rows_up_l.comp > $R/qwen4_mix_rows_up_l.comp
git show fc15cdbf:engine/shaders/qwen4_attention_grouped_dec.glsl > $R/qwen4_attention_grouped_dec-q-lds.glsl
git show fc15cdbf:engine/shaders/qwen4_gdn.comp > $R/qwen4_gdn-unconditional.comp
ls -la $R
