#!/bin/bash
# On incite under a lease: (1) the restructured drafter gives main's draft candidates (replay with the run's tokens,
# x86 paths: SSSE3 products, AVX attention); (2) draft-step timing, main's drafter (instrumented) against the new ones.
cd ~/trashcan-exp/flashnext || exit 1
MODEL=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
mkdir -p results/mtp2-db
for run in water-t1.0 transformer-t0.6; do
    for b in trashcan-mtp2-prof trashcan-mtp2-opt2; do
        ./$b qwen4 mtp replay $MODEL $MTP mtp2-samp/run-$run.json results/mtp2-db/rep-$b-$run.json --depth 3 --window 64 \
            --teacher > /dev/null 2>&1
    done
    python3 -c "
import json
a = json.load(open('results/mtp2-db/rep-trashcan-mtp2-prof-$run.json')); b = json.load(open('results/mtp2-db/rep-trashcan-mtp2-opt2-$run.json'))
print('$run identical draft candidates:', a['draft_tops'] == b['draft_tops'], 'replay s: %.2f -> %.2f' % (a['draft_s'], b['draft_s']))
"
done
for s in 512 3072; do
    for b in trashcan-mtp2-prof trashcan-mtp2-opt1 trashcan-mtp2-opt2 trashcan-mtp2-opt3; do
        bash mtp2-db.sh $b results/mtp2-db/$b $s
    done
done
for s in 2048 4096; do
    for b in trashcan-mtp2-prof trashcan-mtp2-opt2 trashcan-mtp2-opt3; do
        bash mtp2-db.sh $b results/mtp2-db/$b $s
    done
done
