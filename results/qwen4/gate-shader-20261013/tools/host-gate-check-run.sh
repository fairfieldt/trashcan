#!/bin/bash
# Run on alight under a gpuq lease (written by gate-check.sh).
cd ~/trashcan-exp/flashnext || exit 1
for g in 0 1; do
    scripts/with-clocks.sh results/clocks/host-gate1-alight-$g.txt ./trashcan-host-gate1 qwen4 gate-check --gpu $g --reps 200         > results/host-gate1-alight-$g.json 2> results/host-gate1-alight-$g.err
    echo "gpu $g rc $?"
done
scripts/with-clocks.sh results/clocks/host-gate1-alight-0-high.txt ./trashcan-host-gate1 qwen4 gate-check --gpu 0 --reps 200 --high     > results/host-gate1-alight-0-high.json 2> results/host-gate1-alight-0-high.err
echo "gpu 0 high rc $?"
echo "amdgpu timeouts/resets in dmesg: $(sudo -n dmesg | grep -c -i -E 'amdgpu.*(timeout|reset)|gpu reset')"
