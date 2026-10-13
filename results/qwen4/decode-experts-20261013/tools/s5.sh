#!/bin/bash
# Session 5 on enlight (under a lease): where top-k's time goes (return probes after the loads, the maximum, the
# probabilities and the threshold) and a top-k with the per-lane expert count fixed and branch-free compaction.
cd ~/trashcan-exp/flashnext
X=trashcan-moe-x
./moe-sb.sh prof moe-s5-x-b1 $X 1
for f in 1 2 4 8; do ./moe-sb.sh prof moe-s5-tkp$f-b1 $X 1 TRASHCAN_MOE_X=tkp$f; done
./moe-sb.sh prof moe-s5-tk3-b1 $X 1 TRASHCAN_MOE_X=tk3
./moe-sb.sh prof moe-s5-tk3-b3 $X 3 TRASHCAN_MOE_X=tk3
./moe-sb.sh dump moe-s5-dump-tk3-c3 $X 3 dec-ref14.ids TRASHCAN_MOE_X=tk3
./moe-sb.sh dump moe-s5-dump-base-c3 trashcan-6f6635a 3 dec-ref14.ids
./moe-sb.sh prof moe-s5-new-b2 trashcan-moe 2
./moe-sb.sh prof moe-s5-base-b2 trashcan-6f6635a 2
./moe-sb.sh prof moe-s5-base-b4 trashcan-6f6635a 4
echo "moe-s5 done"
