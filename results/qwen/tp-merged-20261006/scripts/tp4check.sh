#!/bin/bash
# Driver side (incite) of the 27B TP4 correctness runs over USB4STREAM, with the node (qwen-tp-node --mtp) already
# running on invite. usage (inside the incite lease): tp4check.sh [BIN]
cd ~/trashcan-exp/merged
bin=${1:-./trashcan}
M=~/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf
MTP=~/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf
S=/dev/tbstream0,/dev/tbstream1,/dev/tbstream2
./run.sh val-27b-tp4 $bin qwen-validate $M results/val-27b-tp4 --tp-stream $S
sleep 2
./run.sh check-27b-tp4 $bin qwen-check $M --tp-stream $S --verify 3 --tokens 600 --steps 128
sleep 2
./run.sh checkmtp-27b-tp4 $bin qwen-check $M --tp-stream $S --mtp $MTP --tokens 600 --steps 256
sleep 2
./run.sh check700-27b-tp4 $bin qwen-check $M --tp-stream $S --verify 3 --tokens 700 --steps 128
sleep 2
./run.sh check700mtp-27b-tp4 $bin qwen-check $M --tp-stream $S --mtp $MTP --tokens 700 --steps 256
sleep 2
./run.sh check4300-27b-tp4 $bin qwen-check $M --tp-stream $S --verify 3 --tokens 4300 --steps 64
for f in check-27b-tp4 checkmtp-27b-tp4 check700-27b-tp4 check700mtp-27b-tp4 check4300-27b-tp4; do
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); s=d['speculative'] or {}; print(sys.argv[1], d['prompt_tokens'], 'pass' if d['pass'] else 'FAIL', d['restore_exact'], d['verify_rounds'], len(d['column_mismatches']), d['max_column_score_error'], round(d['prefill_tok_s'],1), round(d['decode_tok_s'],2), s.get('identical'), (s.get('stats') or {}).get('accepted'), (s.get('stats') or {}).get('proposed'))" results/$f.json
done
grep greedy results/val-27b-tp4.json results/val-27b-tp4.err | tail -2
