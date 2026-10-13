#!/bin/bash
# Session 4 on enlight (under a lease): the rebased final kernels (trashcan-moe on 6f6635a) against trashcan-6f6635a:
# profiles at 1 and 3 tokens, stage dumps of the 14-token prompt in chunks of 1, 3 and 5 (bit-identity), and the
# experimental pair kernels (TRASHCAN_MOE_AB=pairs) at 2-4 tokens.
cd ~/trashcan-exp/flashnext
B=trashcan-6f6635a N=trashcan-moe
./moe-sb.sh prof moe-s4-new-b1 $N 1
./moe-sb.sh prof moe-s4-base-b1 $B 1
./moe-sb.sh prof moe-s4-new-b3 $N 3
./moe-sb.sh prof moe-s4-base-b3 $B 3
for c in 1 3 5; do
    ./moe-sb.sh dump moe-s4-dump-new-c$c $N $c dec-ref14.ids
    ./moe-sb.sh dump moe-s4-dump-base-c$c $B $c dec-ref14.ids
done
./moe-sb.sh dump moe-s4-dump-pairs-c3 $N 3 dec-ref14.ids TRASHCAN_MOE_AB=pairs
./moe-sb.sh dump moe-s4-dump-pairs-c5 $N 5 dec-ref14.ids TRASHCAN_MOE_AB=pairs
./moe-sb.sh prof moe-s4-pairs-b2 $N 2 TRASHCAN_MOE_AB=pairs
./moe-sb.sh prof moe-s4-pairs-b3 $N 3 TRASHCAN_MOE_AB=pairs
./moe-sb.sh prof moe-s4-pairs-b4 $N 4 TRASHCAN_MOE_AB=pairs
./moe-sb.sh prof moe-s4-new-b4 $N 4
./moe-sb.sh prof moe-s4-g2r-b1 $N 1 TRASHCAN_MOE_AB=g2r
./moe-sb.sh prof moe-s4-dwi-b1 $N 1 TRASHCAN_MOE_AB=dwi
./moe-sb.sh prof moe-s4-dwi-b3 $N 3 TRASHCAN_MOE_AB=dwi
./moe-sb.sh prof moe-s4-tk2x-b1 $N 1 TRASHCAN_MOE_AB=tk2x
./moe-sb.sh prof moe-s4-g2r-b3 $N 3 TRASHCAN_MOE_AB=g2r
./moe-sb.sh dump moe-s4-dump-g2r-c3 $N 3 dec-ref14.ids TRASHCAN_MOE_AB=g2r
./moe-sb.sh dump moe-s4-dump-dwi-c3 $N 3 dec-ref14.ids TRASHCAN_MOE_AB=dwi
./moe-sb.sh dump moe-s4-dump-new-p16 $N 16 moe-p100.ids
./moe-sb.sh dump moe-s4-dump-base-p16 $B 16 moe-p100.ids
echo "moe-s4 done"
