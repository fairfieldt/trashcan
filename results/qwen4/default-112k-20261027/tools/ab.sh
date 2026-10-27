#!/bin/bash
# Flash-Next default context A/B (2026-10-27): A = the 6-flashnext recipe until now (32768, f16 K/V), B = the candidate
# (114688, Q8 K/V), same binary (v5 = origin/main's engine). Validation on the 14-token and p100 prompts, decode after
# 512 (ABBA), prefill at 2K, MTP on the chat prompts (one pass each), one 100K-token prompt with B, then the launcher's
# smoke of the edited recipe with --mtp. Under chain6.sh (TICKETS set). DRY=1 prints the runs.
W0=/Users/fairfieldt/dev/claude-work/flashnext-100k-20261027
cd $W0/lab/results/qwen4/chain-20261012 || exit 1
BIN=trashcan-v5
TAG=fn100k
W=/home/fairfieldt/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
eval "$(python3 tools/ring.py)" || exit 1
N=$ADDRS
A="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
B="--gpus 0,1 --nodes $N --head-split 12 --context 114688 --kv q8 --batch 128 --ngram disk"
P=$W/mtp-prompts
run() { # name driver-args [extra q4run args]
    local name=$1 driver=$2; shift 2
    if [ "${DRY:-0}" = 1 ]; then echo "run $TAG-$name: $driver" | cut -c1-170; return; fi
    local s=$(date +%s)
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$BIN" --name "$TAG-$name" --driver-host incite \
        --nodes "$NODES" "$@" --driver "$driver" > /dev/null 2>&1
    echo "$TAG-$name rc $? $(( $(date +%s) - s )) s"
    check_links "$name"
}
check_links() { # name
    local line
    line=$(ssh incite "grep -h '^links: ' $W/results/$TAG-$1.err 2>/dev/null | tail -1")
    if [ -z "$line" ]; then echo "$TAG-$1 links: no report"; return 0; fi
    if ! python3 -c 'import sys
bad = [e for e in sys.argv[1][len("links: "):].split(", ") if " stream " not in e + " "]
bad and print("\n".join("not on its stream: " + e for e in bad)); sys.exit(1 if bad else 0)' "$line"; then
        echo "$TAG-$1: a link was not on its stream; stopping"
        exit 1
    fi
}
run a-val14 "validate results/$TAG-a-val14 $A --greedy 10"
run b-val14 "validate results/$TAG-b-val14 $B --greedy 10"
run a-val100 "validate results/$TAG-a-val100 $A --greedy 10 --tokens-file $W/p100.ids"
run b-val100 "validate results/$TAG-b-val100 $B --greedy 10 --tokens-file $W/p100.ids"
run a-pp512-1 "bench $A --pp 512 --tg 128 --tokens-file $W/long.ids"
run b-pp512-1 "bench $B --pp 512 --tg 128 --tokens-file $W/long.ids"
run b-pp512-2 "bench $B --pp 512 --tg 128 --tokens-file $W/long.ids"
run a-pp512-2 "bench $A --pp 512 --tg 128 --tokens-file $W/long.ids"
run a-pp2048 "bench $A --pp 2048 --tg 32 --tokens-file $W/long.ids"
run b-pp2048 "bench $B --pp 2048 --tg 32 --tokens-file $W/long.ids"
run a-mtp "./$BIN qwen4 mtp spec $M $A --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $P/water.ids $P/csv.ids $P/transformer.ids" --raw
run b-mtp "./$BIN qwen4 mtp spec $M $B --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $P/water.ids $P/csv.ids $P/transformer.ids" --raw
run b-pp100k "bench $B --pp 100000 --tg 32 --tokens-file $W/long-100k.ids"
# The edited recipe through the launcher, as a user starts it (with MTP), under the same leases.
WRAPS=(); for t in ${TICKETS//,/ }; do WRAPS+=(--wrap "${t%%=*}=gpuq run ${t#*=} --"); done
if [ "${DRY:-0}" = 1 ]; then echo "smoke 6-flashnext --mtp ${WRAPS[*]}"; exit 0; fi
mkdir -p $W0/runs
python3 -u $W0/main/deploy/trashcan-cluster -c $W0/cluster.toml smoke 6-flashnext --mtp "${WRAPS[@]}" > $W0/runs/smoke-mtp.log 2>&1
echo "$TAG-smoke-mtp rc $?"
rsync -a "incite:$W/results/$TAG-*" "incite:$W/results/clocks/$TAG-*" $W0/runs/ 2>/dev/null
for h in enlight invite indite alight inflight; do
    rsync -a "$h:$W/results/node-$TAG-*" "$h:$W/results/clocks/*$TAG-*" $W0/runs/ 2>/dev/null
done
echo "fn100k done"
