#!/bin/bash
# usage: bench.sh NAME BINARY ARGS...   (env passed through); writes results/NAME.json and results/NAME.err
name=$1; bin=$2; shift 2
cd ~/trashcan-exp/decode
mkdir -p results/clocks
scripts/with-clocks.sh results/clocks/$name.txt sudo -n --preserve-env=TRASHCAN_TP_PROFILE,TRASHCAN_TP_PROFILE_DISPATCH,TRASHCAN_TP_EVENTS,TRASHCAN_TP_FREE,TRASHCAN_TP_SUBMIT,TRASHCAN_QWEN_ORDER,TRASHCAN_Q4_DECODE,TRASHCAN_QWEN_ATTN,TRASHCAN_QWEN_NORM,TRASHCAN_TP_ARENA,TRASHCAN_TP_PARTS,TRASHCAN_QWEN_MERGE,TRASHCAN_Q4_MULTI,TRASHCAN_TP_HOSTNORM,TRASHCAN_QWEN_TOPK,TRASHCAN_TP_DRAFT_HEAD python3 scripts/latency-run.py $bin "$@" > results/$name.json 2> results/$name.err
rc=$?
python3 - results/$name.json <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(f"{sys.argv[1]}: pp {d['pp_tok_s']:.2f} tg {d['tg_tok_s']:.2f} tp_ms {d.get('tg_tp_ms')} mtp {d.get('mtp') and {k: round(v,3) if isinstance(v,float) else v for k,v in d['mtp'].items() if k in ('acceptance','tokens_per_round','verify_ms_per_round','draft_ms_per_round')}}")
except Exception as e:
    print(sys.argv[1], "failed", e)
PY
grep "\[clocks\]" results/$name.err
exit $rc
