#!/bin/bash
# Session 2 on enlight (under a lease): v3/v4 expert kernels (main's lane layouts, fewer instructions) against main.
cd ~/trashcan-exp/flashnext
B=trashcan-4c1346d N=trashcan-moe
./moe-sb.sh prof moe-s2-new-b1 $N 1
./moe-sb.sh prof moe-s2-base-b1 $B 1
./moe-sb.sh prof moe-s2-g4-b1 $N 1 TRASHCAN_MOE_AB=g4
./moe-sb.sh prof moe-s2-n-b1 $N 1 TRASHCAN_MOE_AB=g3n,d3n
./moe-sb.sh prof moe-s2-d3r2-b1 $N 1 TRASHCAN_MOE_AB=g4,d3r2
./moe-sb.sh prof moe-s2-nosplit-b1 $N 1 TRASHCAN_MOE_AB=nosplit
./moe-sb.sh prof moe-s2-new-b3 $N 3
./moe-sb.sh prof moe-s2-g4-b3 $N 3 TRASHCAN_MOE_AB=g4
./moe-sb.sh prof moe-s2-m2-b3 $N 3 TRASHCAN_MOE_AB=m2,swap
./moe-sb.sh prof moe-s2-swap-b3 $N 3 TRASHCAN_MOE_AB=swap
./moe-sb.sh dump moe-s2-dump-new-b1 $N 1 dec-ref14.ids
./moe-sb.sh dump moe-s2-dump-new-b3 $N 3 dec-ref14.ids
./moe-sb.sh dump moe-s2-dump-g4-b1 $N 1 dec-ref14.ids TRASHCAN_MOE_AB=g4
./moe-sb.sh prof moe-s2-tk0-b1 $N 1 TRASHCAN_MOE_AB=tk0,nosplit
./moe-sb.sh prof moe-s2-tk2-b1 $N 1 TRASHCAN_MOE_AB=tk2,nosplit
./moe-sb.sh prof moe-s2-tkb-b1 $N 1 TRASHCAN_MOE_AB=tkb,nosplit
./moe-sb.sh dump moe-s2-dump-tkb-b1 $N 1 dec-ref14.ids TRASHCAN_MOE_AB=tkb
echo "moe-s2 done"
