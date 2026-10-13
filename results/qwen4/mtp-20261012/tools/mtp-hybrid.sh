#!/bin/bash
# GPU verification batches before the six-Mac chain: incite's GPUs hold layers 0-5 (with the n-gram module), the
# M3 Ultra's CPU (node ms.tdf.sh:7179) layers 6-47 and the head. Greedy with and without drafts must agree.
cd ~/trashcan-exp/flashnext || exit 1
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
T=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
P=/home/fairfieldt/trashcan-exp/flashnext/mtp-prompts
./trashcan-mtp qwen4 mtp spec $M --gpus 0,1 --cuts 3,6 --nodes ms.tdf.sh:7179 --context 512 --batch 64 \
    --mtp $T --draft-tokens 3 --draft-threads 12 --tg 24 > results/mtp-hybrid-ref.json 2> results/mtp-hybrid-ref.err
echo "reference rc $?"
./trashcan-mtp qwen4 mtp spec $M --gpus 0,1 --cuts 3,6 --nodes ms.tdf.sh:7179 --context 512 --batch 64 \
    --mtp $T --draft-tokens 3 --draft-threads 12 --tg 32 $P/csv.ids $P/water.ids > results/mtp-hybrid-text.json \
    2> results/mtp-hybrid-text.err
echo "text rc $?"
python3 -c "
import json
for f in ['results/mtp-hybrid-ref.json', 'results/mtp-hybrid-text.json']:
    for l in open(f):
        r = json.loads(l)
        print(r['prompt'], 'identical', r['identical'], r['first_difference'], 'tok/round %.2f' % r['tokens_per_round'], r['mtp_ids'][:12])
"
