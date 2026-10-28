#!/bin/bash
# The recipe's smoke (with its MTP default) under gpuq leases on all six, as cluster.sh takes them.
set -uo pipefail
C=$HOME/trashcan-exp/cluster
~/trashcan-exp/webchat/cluster.sh down
T=(); for m in incite inflight alight indite invite enlight; do
  id=$(gpuq --on $m request "smoke mtp-latency" --est 10 --owner "trashcan mtp" | sed -n 's/^Ticket #\([0-9]*\) created.*/\1/p'); T+=("$m=$id"); done
for t in "${T[@]}"; do gpuq --on "${t%%=*}" wait "${t#*=}" > /dev/null; done
W=(); for t in "${T[@]}"; do W+=(--wrap "${t%%=*}=gpuq run ${t#*=} --"); done
python3 -u $C/deploy/trashcan-cluster -c $C/cluster.toml smoke 6-flashnext --nodes incite,inflight,alight,indite,invite,enlight "${W[@]}" 2>&1 | tail -6
rc=${PIPESTATUS[0]}
for t in "${T[@]}"; do gpuq --on "${t%%=*}" release "${t#*=}" -r "smoke rc $rc" > /dev/null 2>&1; done
exit $rc
