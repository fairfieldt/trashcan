#!/bin/bash
# Session 6 on enlight (under a lease): the shorter top-k (8 keys per lane fixed, the threshold loop rolled,
# compaction without branches) in trashcan-moe2 against trashcan-moe at 1 and 3 tokens, and stage dumps in chunks of
# 1, 3, 5 and 16 (to compare with session 4's dumps of trashcan-moe: the same bits expected).
cd ~/trashcan-exp/flashnext
N2=trashcan-moe2
./moe-sb.sh prof moe-s6-tk4-b1 $N2 1
./moe-sb.sh prof moe-s6-new-b1 trashcan-moe 1
./moe-sb.sh prof moe-s6-tk4-b3 $N2 3
./moe-sb.sh prof moe-s6-new-b3 trashcan-moe 3
./moe-sb.sh dump moe-s6-dump-tk4-c1 $N2 1 dec-ref14.ids
./moe-sb.sh dump moe-s6-dump-tk4-c3 $N2 3 dec-ref14.ids
./moe-sb.sh dump moe-s6-dump-tk4-c5 $N2 5 dec-ref14.ids
./moe-sb.sh dump moe-s6-dump-tk4-p16 $N2 16 moe-p100.ids
echo "moe-s6 done"
