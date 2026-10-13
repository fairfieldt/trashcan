#!/bin/bash
# Session 8 on enlight (under a lease): the branch rebased on main 7da6f93 (trashcan-moe3) against trashcan-7da6f93,
# stage dumps of the 14-token prompt in chunks of 1 and 3 (the same bits expected) and of the 100-token prompt in
# chunks of 16 (the router's summation order), and profiles at 1 and 3 tokens.
cd ~/trashcan-exp/flashnext
for c in 1 3; do
    ./moe-sb.sh dump moe-s8-dump-new-c$c trashcan-moe3 $c dec-ref14.ids
    ./moe-sb.sh dump moe-s8-dump-base-c$c trashcan-7da6f93 $c dec-ref14.ids
done
./moe-sb.sh dump moe-s8-dump-new-p16 trashcan-moe3 16 moe-p100.ids
./moe-sb.sh dump moe-s8-dump-base-p16 trashcan-7da6f93 16 moe-p100.ids
./moe-sb.sh prof moe-s8-base-b1 trashcan-7da6f93 1
./moe-sb.sh prof moe-s8-new-b1 trashcan-moe3 1
./moe-sb.sh prof moe-s8-base-b3 trashcan-7da6f93 3
./moe-sb.sh prof moe-s8-new-b3 trashcan-moe3 3
echo "moe-s8 done"
