#!/bin/bash
# On incite under a lease: draft-step timing, main 4c1346d against this branch (opt4: restructured drafter, AVX
# attention, rounding without libm), alternating; then the instrumented opt4 at 512 and 3072.
cd ~/trashcan-exp/flashnext || exit 1
for s in 512 2048 3072 4096; do
    for b in trashcan-4c1346d trashcan-mtp2-opt4; do
        bash mtp2-db.sh $b results/mtp2-db2/$b $s
    done
done
for s in 512 3072; do
    bash mtp2-db.sh trashcan-mtp2-opt4p results/mtp2-db2/trashcan-mtp2-opt4p $s
done
