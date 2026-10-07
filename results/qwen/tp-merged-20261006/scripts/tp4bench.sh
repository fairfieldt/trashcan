#!/bin/bash
# Driver side (incite) of the 27B TP4 benchmarks over USB4STREAM, with the node (qwen-tp-node --mtp) running on
# invite; performance governor and the 0 us CPU latency request on both Macs. usage: tp4bench.sh PREFIX [BIN]
cd ~/trashcan-exp/merged
p=$1; bin=${2:-./trashcan}
M=~/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf
MTP=~/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf
S=/dev/tbstream0,/dev/tbstream1,/dev/tbstream2
for r in a b; do
    ./run.sh $p-27b-tp4-$r $bin qwen-bench $M --tp-stream $S --pp 512 --tg 128; sleep 2
done
for r in a b; do
    ./run.sh $p-27b-tp4-mtp-$r $bin qwen-bench $M --tp-stream $S --pp 512 --tg 128 --mtp $MTP; sleep 2
done
for r in a b; do
    ./run.sh $p-27b-tp4-pp2048-$r $bin qwen-bench $M --tp-stream $S --pp 2048 --tg 32; sleep 2
done
for f in results/$p-27b-tp4-*.json; do
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); m=d.get('mtp') or {}; print(sys.argv[1], round(d['pp_tok_s'],2), round(d['tg_tok_s'],2), m and round(m['acceptance'],3), m and round(m['verify_ms_per_round'],2), {k: round(v,2) for k,v in (d.get('tg_tp_ms') or {}).items() if k in ('wait','exchange','add')})" $f
done
