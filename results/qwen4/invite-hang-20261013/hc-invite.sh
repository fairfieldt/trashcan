#!/bin/bash
# invite health check after the 2026-10-13 15:35 hang: main binary, invite's chain layers (32-39) on both GPUs,
# decode and prefill at four context sizes, three rounds; GPU and PCIe errors counted after every run.
cd ~/trashcan-exp/flashnext || exit 1
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
errs() { sudo -n dmesg | grep -c -i -E "amdgpu.*(timeout|reset|fault|error)|AER|CmpltTO|device inaccessible"; }
echo "start $(date +%T) errors $(errs)"
for r in 1 2 3; do
    for s in 64 3072 8192 30080; do
        name=hc-invite-$r-$s
        scripts/with-clocks.sh results/clocks/$name.txt ./trashcan-5e48d7e qwen4 stage-bench $M --ranges 32:36,36:40 \
            --gpus 0,1 --context 32768 --batch 128 --prefill 3 --decode 32 --start $s > results/$name.json 2> results/$name.err
        echo "$(date +%T) $name rc=$? errors $(errs) $(grep -o "\[clocks\].*max [0-9]*C" results/$name.err | tail -1 | cut -c1-120)"
    done
done
echo "done $(date +%T)"
