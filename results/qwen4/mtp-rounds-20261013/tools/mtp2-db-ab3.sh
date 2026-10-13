#!/bin/bash
# On incite under a lease: draft-step timing, main 4c1346d against the session binary (23135cb), alternating.
cd ~/trashcan-exp/flashnext || exit 1
for s in 512 3072; do
    for b in trashcan-4c1346d trashcan-mtp2; do
        bash mtp2-db.sh $b results/mtp2-db3/$b $s
    done
done
