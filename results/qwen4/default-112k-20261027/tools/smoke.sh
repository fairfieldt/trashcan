#!/bin/bash
# The edited 6-flashnext recipe through the launcher, with MTP, under chain6.sh's leases (TICKETS), while fan-watch.sh
# logs every Mac's fan every 2 s.
W0=/Users/fairfieldt/dev/claude-work/flashnext-100k-20261027
$W0/fan-watch.sh $W0/fan/smoke-fans.log 5 2 &
watch=$!
sleep 6
WRAPS=(); for t in ${TICKETS//,/ }; do WRAPS+=(--wrap "${t%%=*}=gpuq run ${t#*=} --"); done
echo "$(date +%T) smoke starts" >> $W0/fan/smoke-fans.log
python3 -u $W0/main/deploy/trashcan-cluster -c $W0/cluster.toml smoke 6-flashnext --mtp --nodes incite,inflight,alight,indite,invite,enlight "${WRAPS[@]}" > $W0/runs/smoke-mtp.log 2>&1
rc=$?
echo "$(date +%T) smoke ends rc $rc" >> $W0/fan/smoke-fans.log
kill $watch 2>/dev/null
echo "smoke rc $rc"
