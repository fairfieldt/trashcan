#!/bin/bash
# ABBA on incite: A = MTP without the driver's latency request, B = with it (mtp_cpu_latency), then plain (no MTP).
cd ~/trashcan-exp/mtp-latency-20261028
R=~/trashcan-exp/cluster/deploy/recipes.toml
AB=~/trashcan-exp/demo-check-20261028/ab.py
CSV="Write a CSV of all 50 US states with columns state,abbreviation,capital."
run() {   # label, latency true/false, switcher
    sed -i "s/^mtp_cpu_latency = .*/mtp_cpu_latency = $2/" $R
    $3 up 6-flashnext > up-$1.log 2>&1 || { echo "$1: start failed"; tail -3 up-$1.log; exit 1; }
    echo "== $1 (mtp_cpu_latency=$2, latency holders: $(pgrep -f 'scripts/latency-run.py' | wc -l))"
    for i in 1 2; do python3 $AB default | tail -1; python3 $AB 0 "$CSV" none | tail -1; done
}
run A1 false ~/trashcan-exp/webchat/cluster.sh
run B1 true ~/trashcan-exp/webchat/cluster.sh
run B2 true ~/trashcan-exp/webchat/cluster.sh
run A2 false ~/trashcan-exp/webchat/cluster.sh
run P true ~/trashcan-exp/demo-check-20261028/cluster-nomtp.sh
sed -i "s/^mtp_cpu_latency = .*/mtp_cpu_latency = true/" $R
