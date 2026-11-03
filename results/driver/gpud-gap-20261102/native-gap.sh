#!/bin/bash
# Native comparison runs for gpud's remaining gap, tagged with $1 (the boot entry): norm micro-benchmark, a
# dispatch-profiled TP2 decode and the TP2 bench, all on amdgpu. Run under the incite lease.
set -u
cd ~/trashcan-exp/gpud
tag=$1
M=~/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf
echo "== $tag: $(cat /proc/cmdline | grep -oE 'intel_iommu=[a-z]+|iommu=pt|amdgpu.vis_vramlimit=[0-9]+' | tr '\n' ' ')"
dmesg | grep -iE "visible|vram.*limit|DMAR: .*translat|iommu.*default domain" | head -3
engine/trashcan norm-bench > nb-$tag.txt 2>&1
grep "^{" nb-$tag.txt | python3 -I -c 'import json,sys; print("norm-bench", [round(json.loads(l)["median_us"],2) for l in sys.stdin][:6])'
rm -f prof-$tag.jsonl
TRASHCAN_TP_PROFILE=prof-$tag.jsonl TRASHCAN_TP_PROFILE_DISPATCH=1 engine/trashcan qwen-bench $M --tp --pp 512 --tg 64 > benchprof-$tag.txt 2>&1
python3 -I tp_profile.py prof-$tag.jsonl --dispatch 2>&1 | grep -E "gpu segments|headvec|attention_merge|attention_partial|qwen_gdn|qwen_matf32|qwen_conv|matvecq4_r2u2 2048" | head -8
scripts/with-clocks.sh clocks-$tag.txt engine/trashcan qwen-bench $M --tp --pp 512 --tg 128 --reps 2 > bench-$tag.txt 2>&1
grep "^{" bench-$tag.txt | python3 -I -c '
import json,sys
for l in sys.stdin:
    j=json.loads(l); print("TP2 pp512 %.1f tok/s, tg128 %.1f tok/s" % (j["pp_tok_s"], j["tg_tok_s"]))'
