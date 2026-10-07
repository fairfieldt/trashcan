#!/bin/bash
# From the Mac: 27B TP4 A/B over USB4STREAM, the node on invite restarted with the driver's binary before each run.
# usage: ab27b.sh INCITE_LEASE INVITE_LEASE NAME BIN NODE_ENV QWEN_BENCH_ARGS...
#   BIN: ./trashcan, ./trashcan-decode or ./trashcan-prefill; NODE_ENV "nolat" starts the node without the
#   0 us CPU latency request (as the prefill branch's runs did)
li=$1; lv=$2; name=$3; bin=$4; nodeenv=$5; shift 5
M='~/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf'
MTP='~/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf'
S=/dev/tbstream0,/dev/tbstream1,/dev/tbstream2
if [ "$nodeenv" = nolat ]; then
    start="nohup scripts/with-clocks.sh results/clocks/node-$name.txt $bin qwen-tp-node $M --mtp $MTP --stream $S > results/node-$name.log 2>&1 < /dev/null &"
else
    start="BIN=$bin ./node.sh node-$name $M --mtp $MTP"
fi
ssh invite "cd ~/trashcan-exp/merged && pkill -f '^[.]/trashcan[-a-z]* qwen-tp-node'; sleep 4; echo \"\$(date -Is) node-$name ($nodeenv): $bin\" >> results/commands.log; gpuq run $lv -- bash -c '$start'; sleep 2" > /dev/null
ssh incite "cd ~/trashcan-exp/merged && gpuq run $li -- ./run.sh $name $bin qwen-bench $M --tp-stream $S $*" | grep -v '^\[clocks\]'
ssh incite "cd ~/trashcan-exp/merged && python3 -c \"import json,sys; d=json.load(open(sys.argv[1])); m=d.get('mtp') or {}; print(sys.argv[1], round(d['pp_tok_s'],2), round(d['tg_tok_s'],2), m and round(m['verify_ms_per_round'],2), m and round(m['draft_ms_per_round'],2), {k: round(v,2) for k,v in (d.get('tg_tp_ms') or {}).items() if k in ('wait','exchange','add')})\" results/$name.json"
