#!/bin/bash
# Session 1 on enlight (under a lease): baseline against the first kernels, profiles and 14-token dumps.
cd ~/trashcan-exp/flashnext
B=trashcan-4c1346d N=trashcan-moe
./moe-sb.sh prof moe-s1-base-b1 $B 1
./moe-sb.sh prof moe-s1-new-b1 $N 1
./moe-sb.sh dump moe-s1-dump-base-b1 $B 1 dec-ref14.ids
./moe-sb.sh dump moe-s1-dump-new-b1 $N 1 dec-ref14.ids
./moe-sb.sh prof moe-s1-v2d2-b1 $N 1 TRASHCAN_MOE_AB=v2,d2
./moe-sb.sh prof moe-s1-p4-b1 $N 1 TRASHCAN_MOE_AB=p4
./moe-sb.sh prof moe-s1-r16-b1 $N 1 TRASHCAN_MOE_AB=r16
./moe-sb.sh prof moe-s1-base-b3 $B 3
./moe-sb.sh prof moe-s1-new-b3 $N 3
./moe-sb.sh prof moe-s1-v2d2-b3 $N 3 TRASHCAN_MOE_AB=v2,d2
./moe-sb.sh dump moe-s1-dump-base-b3 $B 3 dec-ref14.ids
./moe-sb.sh dump moe-s1-dump-new-b3 $N 3 dec-ref14.ids
./moe-sb.sh dump moe-s1-dump-v2d2-b3 $N 3 dec-ref14.ids TRASHCAN_MOE_AB=v2,d2
./moe-sb.sh dump moe-s1-dump-v2d2-b1 $N 1 dec-ref14.ids TRASHCAN_MOE_AB=v2,d2
echo "moe-s1 done"
