#!/bin/bash
# Session 9 on invite (one-Mac lease, ~7 min): this branch rebased on main 4307169 (small-batch experts) as
# trashcan-vfy10 against trashcan-4307169: stage-bench at batches 2-4 with single-token decode, and 96-token dumps.
cd ~/trashcan-exp/flashnext
B=trashcan-4307169 V=trashcan-vfy10
./vfy-sb.sh 64 16 8 vfy-s9-base-b3=$B:3 vfy-s9-new-b3=$V:3 vfy-s9-new-b2=$V:2 vfy-s9-base-b2=$B:2 vfy-s9-new-b4=$V:4 \
    vfy-s9-base-b4=$B:4
./vfy-dump.sh vfy-p96.ids vfy-s9-dnew-b1=$V:1 vfy-s9-dnew-b4=$V:4 vfy-s9-dbase-b1=$B:1
echo "session 9 done"
