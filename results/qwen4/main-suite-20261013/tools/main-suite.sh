#!/bin/bash
# Main's numbers, measured once per main commit for every agent (one run each; noise is about 1%): validation on the
# 14-token reference prompt and p100, decode after a 512-token prompt (two runs), prefill at 2K, 8K and 30K tokens, MTP on
# the chat prompts (one pass, plain and speculative), and the server test. Under chain6.sh (TICKETS set); BIN is the
# binary on every Mac (default trashcan-4307169). DRY=1 prints the runs instead.
cd /Users/fairfieldt/dev/trashcan/results/qwen4/chain-20261012 || exit 1
BIN=${BIN:-trashcan-4307169}
TAG=${TAG:-ms-${BIN#trashcan-}}
W=/home/fairfieldt/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
# Chain order and node addresses from the live Thunderbolt links (tools/ring.py), so recabling needs no edit here.
eval "$(python3 tools/ring.py)" || exit 1
N=$ADDRS
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
P=$W/mtp-prompts
run() { # name driver-args [extra q4run args]
    local name=$1 driver=$2; shift 2
    if [ "${DRY:-0}" = 1 ]; then echo "run $TAG-$name: $driver" | cut -c1-150; return; fi
    local s=$(date +%s)
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$BIN" --name "$TAG-$name" --driver-host incite \
        --nodes "$NODES" "$@" --driver "$driver" > /dev/null 2>&1
    echo "$TAG-$name rc $? $(( $(date +%s) - s )) s"
}
run val14 "validate results/$TAG-val14 $C --greedy 10"
run val100 "validate results/$TAG-val100 $C --greedy 10 --tokens-file $W/p100.ids"
run pp512-1 "bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
run pp512-2 "bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
run pp2048 "bench $C --pp 2048 --tg 32 --tokens-file $W/long.ids"
run pp8192 "bench $C --pp 8192 --tg 32 --tokens-file $W/long-32k.ids"
run pp30000 "bench $C --pp 30000 --tg 32 --tokens-file $W/long-32k.ids"
run mtp "./$BIN qwen4 mtp spec $M $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $P/water.ids $P/csv.ids $P/transformer.ids" --raw
run serve "bash scripts/q4serve-test.sh $M results/$TAG-serve-server.log --nodes $N --head-split 12 --context 16384 --batch 128 --ngram disk --mtp $MTP --draft-tokens 3" --raw --env "BIN=./$BIN"
[ "${DRY:-0}" = 1 ] && exit 0
mkdir -p /Users/fairfieldt/dev/trashcan/results/qwen4/main-suite-20261013/$TAG
rsync -a "incite:$W/results/$TAG-*" /Users/fairfieldt/dev/trashcan/results/qwen4/main-suite-20261013/$TAG/
echo "main suite $TAG done"
