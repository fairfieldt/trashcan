#!/bin/bash
# Final drafter on incite (4-bit dense, Q8 head): threads, the CPU's share with the GPUs scoring rows 0-43007, then
# the GPU-head path end to end (incite's GPUs hold layers 0-5 and two thirds of the head, the M3 Ultra layers 6-47).
cd ~/trashcan-exp/flashnext || exit 1
B=./trashcan-mtp5
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
T=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
P=/home/fairfieldt/trashcan-exp/flashnext/mtp-prompts
O=results/mtp-cpu6
mkdir -p $O
run() {
    name=$1 threads=$2 vocab=$3 skip=$4
    sudo -n python3 scripts/latency-run.py $B qwen4 mtp draft-bench $M $T --context 8192 --start 512 --steps 64 \
        --batch 1,2 --threads $threads --vocab $vocab --skip-head-rows $skip > $O/$name.json 2> $O/$name.err
    echo "$name: $(python3 -c "
import json; r=json.load(open('$O/$name.json'))
print('MiB %.0f' % r['step_mib'], 'prefill %.0f tok/s' % r['prefill_tok_s'], ' '.join('b%d %.2f/%.2f' % (s['batch'], s['p50_ms'], s['min_ms']) for s in r['steps']),
      '| b1', ' '.join('%s %.2f' % (k[:9], v) for k, v in r['steps'][0]['phases_ms'].items()))")"
}
run t12 12 65536 0
run t24 24 65536 0
run t12-skip43008 12 65536 43008
run t24-skip43008 24 65536 43008
run t12-v98304-skip43008 12 98304 43008
$B qwen4 mtp spec $M --gpus 0,1 --cuts 3,6 --nodes ms.tdf.sh:7179 --head-split 3 --context 512 --batch 64 \
    --mtp $T --draft-tokens 3 --draft-threads 12 --tg 32 $P/csv.ids $P/water.ids > $O/hybrid.json 2> $O/hybrid.err
echo "hybrid rc $?"
python3 -c "
import json
for l in open('$O/hybrid.json'):
    r = json.loads(l)
    print(r['prompt'], 'identical', r['identical'], r['first_difference'], 'tok/round %.2f' % r['tokens_per_round'], 'draft ms/round %.1f' % r['ms_per_round']['draft'])
"
