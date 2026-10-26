#!/bin/bash
# Repeated 6-flashnext smokes under the leases in $TICKETS, with per-node monitors (vmstat, PSI) during each run.
# usage: TICKETS=incite=ID,... smokes.sh N LAUNCHER OUTDIR [extra launcher args]
set -uo pipefail
n=$1 launcher=$2 out=$3; shift 3
inv=$HOME/dev/claude-work/trashcan-release-20261026/fallback/inv/cluster.toml
mkdir -p "$out"
wraps=()
IFS=, read -ra ts <<< "$TICKETS"
for t in "${ts[@]}"; do wraps+=(--wrap "${t%%=*}=gpuq run ${t#*=} --"); done
hosts="incite invite indite enlight inflight alight"
pass=0 fail=0
for i in $(seq "$n"); do
    tag=$(date +%H%M%S)-$i
    if [ "${COLD:-0}" = 1 ]; then
        for h in $hosts; do ssh "$h" "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null" & done; wait
    fi
    for h in $hosts; do
        ssh "$h" "setsid -f ~/trashcan-exp/fallback/scripts/mon.sh ~/trashcan-exp/fallback/mon/$tag > /dev/null 2>&1 < /dev/null"
    done
    "$launcher" -c "$inv" smoke 6-flashnext --nodes incite,enlight,invite,indite,alight,inflight "${wraps[@]}" "$@" > "$out/smoke-$tag.log" 2>&1
    rc=$?
    for h in $hosts; do ssh "$h" "pkill -f scripts/mon.sh; true" > /dev/null 2>&1; done
    ans=$(grep -o "answer '[^']*'" "$out/smoke-$tag.log" | head -1)
    logs=$(grep -o "logs in .*" "$out/smoke-$tag.log" | sed 's/logs in //')
    ev=""
    if [ -n "$logs" ]; then
        ev=$(grep -hE "stream lost|continuing over TCP|not up within|no turn for|a turn took|link: no stream|not opened|session ended" "$logs"/*.log | grep -v "^$" | head -20)
        cp -r "$logs" "$out/" 2>/dev/null
    fi
    if [ $rc -eq 0 ]; then pass=$((pass+1)); else fail=$((fail+1)); fi
    echo "run $i ($tag): rc $rc $ans"
    [ -n "$ev" ] && echo "$ev" | sed 's/^/    /'
done
echo "smokes: $pass pass, $fail fail of $n"
