#!/bin/bash
# The checks that first ran on the pre-split build (f71f25cf), again with the committed tree's binary: the Gemma
# checks, the 9B pipeline over both GPUs, and the 9B TP2 validate and qwen-check runs. usage (inside the lease):
# final-checks-incite.sh
cd ~/trashcan-exp/merged
cp trashcan-final trashcan
M=~/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf
MTP=~/models/qwen3.5-9b/mtp-Qwen3.5-9B-trashcan-Q4_0.gguf
scripts/with-clocks.sh results/clocks/gemma-test-final.txt ./gemma-test.sh > results/gemma-test-final.log 2>&1
echo "gemma rc=$?"
rm -rf results/val-9b-pipe-final results/val-9b-tp2-final
./run.sh val-9b-pipe-final ./trashcan qwen-validate $M results/val-9b-pipe-final --gpus 0,1
./run.sh val-9b-tp2-final ./trashcan qwen-validate $M results/val-9b-tp2-final --tp
./run.sh check-9b-tp2-final ./trashcan qwen-check $M --tp --verify 3 --tokens 600 --steps 128
./run.sh checkmtp-9b-tp2-final ./trashcan qwen-check $M --tp --mtp $MTP --tokens 600 --steps 256
./run.sh check700-9b-tp2-final ./trashcan qwen-check $M --tp --verify 3 --tokens 700 --steps 128
./run.sh check700mtp-9b-tp2-final ./trashcan qwen-check $M --tp --mtp $MTP --tokens 700 --steps 256
./run.sh check4300-9b-tp2-final ./trashcan qwen-check $M --tp --verify 3 --tokens 4300 --steps 128
for f in check-9b-tp2-final checkmtp-9b-tp2-final check700-9b-tp2-final check700mtp-9b-tp2-final check4300-9b-tp2-final; do
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); s=d['speculative'] or {}; print(sys.argv[1], d['prompt_tokens'], 'pass' if d['pass'] else 'FAIL', d['restore_exact'], d['verify_rounds'], len(d['column_mismatches']), d['max_column_score_error'], round(d['prefill_tok_s'],1), round(d['decode_tok_s'],2), s.get('identical'), (s.get('stats') or {}).get('accepted'), (s.get('stats') or {}).get('proposed'))" results/$f.json
done
grep -h "PASS\|passed" results/gemma-test-final.log | sort | uniq -c
