#!/bin/bash
# Session 7 on enlight (under a lease): the final binary (trashcan-moe2) against main's at 1, 2, 3, 4 and 16 tokens,
# alternated in one session.
cd ~/trashcan-exp/flashnext
for b in 1 2 3 4 16; do
    ./moe-sb.sh prof moe-s7-base-b$b trashcan-6f6635a $b
    ./moe-sb.sh prof moe-s7-new-b$b trashcan-moe2 $b
done
echo "moe-s7 done"
