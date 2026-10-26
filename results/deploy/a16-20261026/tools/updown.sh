#!/bin/bash
# Lab only (A16 acceptance): up, status, a second up (must be refused), down, status.
# usage (via lease.sh): updown.sh RECIPE NODES [ARGS...] -- --wrap ...
set -u
recipe=$1 nodes=$2; shift 2
extra=(); while [ $# -gt 0 ] && [ "$1" != --wrap ]; do extra+=("$1"); shift; done
wraps=("$@")
T=../main/deploy/trashcan-cluster; C=(-c cluster.toml)
$T "${C[@]}" up $recipe --nodes $nodes ${extra[@]+"${extra[@]}"} "${wraps[@]}"; echo "rc $?"
$T "${C[@]}" status
echo "--- a second up while it runs"
$T "${C[@]}" up $recipe --nodes $nodes ${extra[@]+"${extra[@]}"} "${wraps[@]}" 2>&1 | grep -E 'PROBLEM|preflight|error'; echo "rc ${PIPESTATUS[0]}"
$T "${C[@]}" down "${wraps[@]}"; echo "rc $?"
$T "${C[@]}" status
for h in ${nodes//,/ }; do
    echo "$h: engines [$(ssh $h "pgrep -af 'trashcan-exp/a16/trashcan' | grep -v pgrep" | tr '\n' ';')] ports [$(ssh $h "ss -ltnH '( sport = :8000 or sport = :7071 or sport = :7072 )'" | awk '{print $4}' | tr '\n' ' ')] dpm [$(ssh $h 'cat /sys/class/drm/card*/device/power_dpm_force_performance_level' | tr '\n' ' ')]"
done
