#!/bin/bash
# Diagnostic after invite's second GPU incident (GPU1 compute-ring timeout at 16:05, main adf8cfd, pp512 bench): main's
# binary, the failing run's workload repeated, first with the chain reversed (invite holds layers 16-23 and inflight
# 32-39), then in the usual order. Each run has a 3 min limit; after each run every Mac's kernel log is checked for GPU
# or PCIe errors, and the first error stops the session. Under chain6.sh. DRY=1 lists the runs.
cd /Users/fairfieldt/dev/trashcan/results/qwen4/chain-20261012 || exit 1
BIN=${BIN:-trashcan-2293550}
W=/home/fairfieldt/trashcan-exp/flashnext
mkdir -p "$HOME/dev/claude-work/flashnext-20261012/stress-q4run"
ERR='amdgpu.*(timeout|reset|fault)|CmpltTO|device inaccessible|AER: (Uncorrect|Corrected)'
base=$(for h in incite indite inflight alight invite enlight; do echo "$h=$(ssh $h "sudo -n dmesg | grep -c -i -E '$ERR'")"; done | tr '\n' ' ')
echo "start $(date +%T) errors: $base"
errs() { for h in incite indite inflight alight invite enlight; do echo "$h=$(ssh $h "sudo -n dmesg | grep -c -i -E '$ERR'")"; done | tr '\n' ' '; }
run() { # name order driver
    local name=$1 order=$2 driver=$3 nodes addrs
    if [ "$order" = rev ]; then
        nodes=enlight,invite,alight,inflight,indite
        addrs=enlight.tb1:7071,invite.tb0:7071,alight.tb1:7071,inflight.tb0:7071,indite.tb1:7071
    else
        nodes=indite,inflight,alight,invite,enlight
        addrs=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
    fi
    local C="--gpus 0,1 --nodes $addrs --head-split 12 --context 32768 --batch 128 --ngram disk"
    if [ "${DRY:-0}" = 1 ]; then echo "run $name ($order): ${driver/C/$C}" | cut -c1-140; return 0; fi
    local s=$(date +%s)
    timeout 180 python3 tools/q4run.py --tickets "$TICKETS" --binary "$BIN" --name "$name" --driver-host incite \
        --nodes "$nodes" --driver "${driver/C/$C}" > /dev/null 2> "$HOME/dev/claude-work/flashnext-20261012/stress-q4run/$name.err"
    local rc=$? now=$(errs)
    echo "$(date +%T) $name rc $rc $(( $(date +%s) - s )) s $(ssh incite "grep -o '\"\(pp\|tg\)_tok_s\":[0-9.]*' $W/results/$name.json" 2>/dev/null | tr '\n' ' ')"
    if [ "$now" != "$base" ] || [ $rc -ne 0 ]; then echo "STOP: rc $rc, errors now: $now"; exit 1; fi
}
for order in rev fwd; do
    for i in $(seq 1 12); do
        run ${TAG:-st}-$order-pp512-$i $order "bench C --pp 512 --tg 128 --tokens-file $W/long.ids"
        [ $((i % 4)) = 0 ] && run ${TAG:-st}-$order-pp4096-$i $order "bench C --pp 4096 --tg 32 --tokens-file $W/long-32k.ids"
    done
    run ${TAG:-st}-$order-val3061 $order "validate results/${TAG:-st}-$order-val3061 C --greedy 10 --tokens-file $W/long.ids"
done
echo "stress done: no GPU or PCIe errors"
