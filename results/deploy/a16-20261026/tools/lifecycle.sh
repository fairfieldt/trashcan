#!/bin/bash
# Lab only (A16 acceptance): lifecycle checks of deploy/trashcan-cluster for one recipe under leases.
# usage (via lease.sh, which appends --wrap args): lifecycle.sh RECIPE NODES [--mtp] [--tcp-fallback] -- --wrap ...
set -u
recipe=$1 nodes=$2; shift 2
extra=()
while [ $# -gt 0 ] && [ "$1" != --wrap ]; do extra+=("$1"); shift; done
wraps=("$@")
T=../main/deploy/trashcan-cluster
C=(-c cluster.toml)
check() {
    echo "--- after $1: engine processes, ports, DPM on ${nodes}"
    for h in ${nodes//,/ }; do
        echo "$h: engines [$(ssh $h "pgrep -af 'trashcan-exp/a16/trashcan' | grep -v pgrep" | tr '\n' ';')] ports [$(ssh $h "ss -ltnH '( sport = :8000 or sport = :7071 or sport = :7072 )'" | awk '{print $4}' | tr '\n' ' ')] dpm [$(ssh $h 'cat /sys/class/drm/card*/device/power_dpm_force_performance_level' | tr '\n' ' ')]"
    done
}
wait_ready() { until grep -q "ready after" "$1" || ! kill -0 $2 2>/dev/null; do sleep 2; done; }

echo "=== 1. smoke"
$T "${C[@]}" smoke $recipe --nodes $nodes ${extra[@]+"${extra[@]}"} "${wraps[@]}"; echo "rc $?"
check smoke

echo "=== 2. run, then a node process killed (SIGKILL)"
log=$(mktemp "$PWD/state/run.XXXXXX")
$T "${C[@]}" run $recipe --nodes $nodes ${extra[@]+"${extra[@]}"} "${wraps[@]}" > "$log" 2>&1 &
pid=$!
wait_ready "$log" $pid
victim=$(python3 -c "import json; r=json.load(open('state/run.json')); p=[p for p in r['procs'] if p['role']!='driver'][0]; print(p['node'], p['pidfile'])")
echo "killing ${victim% *}'s node process at $(date +%T)"
ssh ${victim% *} "kill -9 \$(cat ${victim#* })"
t0=$(date +%s); wait $pid; rc=$?
cat "$log"; echo "rc $rc, launcher ended $(( $(date +%s) - t0 )) s after the kill"
check "node kill"

echo "=== 3. run, then SIGINT to the launcher"
$T "${C[@]}" run $recipe --nodes $nodes ${extra[@]+"${extra[@]}"} "${wraps[@]}" > "$log" 2>&1 &
pid=$!
wait_ready "$log" $pid
echo "SIGINT at $(date +%T)"; kill -INT $pid
t0=$(date +%s); wait $pid; rc=$?
cat "$log"; echo "rc $rc, stopped $(( $(date +%s) - t0 )) s after SIGINT"
check SIGINT

echo "=== 4. up, status, a second up (refused), down, status"
$T "${C[@]}" up $recipe --nodes $nodes ${extra[@]+"${extra[@]}"} "${wraps[@]}"; echo "rc $?"
$T "${C[@]}" status
$T "${C[@]}" up $recipe --nodes $nodes ${extra[@]+"${extra[@]}"} "${wraps[@]}" 2>&1 | tail -3; echo "rc ${PIPESTATUS[0]}"
$T "${C[@]}" down "${wraps[@]}"; echo "rc $?"
$T "${C[@]}" status
check down
rm -f "$log"
