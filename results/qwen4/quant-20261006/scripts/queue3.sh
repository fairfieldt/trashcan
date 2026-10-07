#!/bin/bash
# After queue 2: the six-Mac candidate G-iq4 (kept for GPU validation), a 3-bit gate/up candidate (five Macs, or
# four with host-RAM experts), then MXFP4 down-projection variants for tighter budgets.
cd /Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash || exit 1
# queue2.sh execs "kld_queue.sh A-hc ...", so watch for both names.
while pgrep -f "queue2.sh" > /dev/null || pgrep -f "kld_queue.sh A-hc" > /dev/null \
    || pgrep -f "make_g_iq4.sh" > /dev/null; do sleep 30; done
if [ -s /Volumes/Green/Qwen/flash-next/cand/G-iq4/G-iq4-00006-of-00006.gguf ]; then
    KEEP=1 ./kld_queue.sh G-iq4 ""
fi
exec ./kld_queue.sh \
  E-iq3xxs "--tensor-type ffn_gate_exps=iq3_xxs --tensor-type ffn_up_exps=iq3_xxs --tensor-type ffn_down_exps=iq4_nl" \
  E-iq4mx "--tensor-type ffn_gate_exps=iq4_xs --tensor-type ffn_up_exps=iq4_xs --tensor-type ffn_down_exps=mxfp4" \
  E-iq2mx "--tensor-type ffn_gate_exps=iq2_xxs --tensor-type ffn_up_exps=iq2_xxs --tensor-type ffn_down_exps=mxfp4"
