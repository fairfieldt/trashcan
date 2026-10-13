#!/bin/bash
# Session 8 on invite (one-Mac lease, ~7 min): this branch rebased on main 7da6f93 with the out projections' K split
# (trashcan-vfy9) against trashcan-7da6f93: stage-bench at batches 2-4 with single-token decode, and 96-token dumps.
cd ~/trashcan-exp/flashnext
B=trashcan-7da6f93 V=trashcan-vfy9
./vfy-sb.sh 64 16 8 vfy-s8-base-b3=$B:3 vfy-s8-new-b3=$V:3 vfy-s8-new-b2=$V:2 vfy-s8-base-b2=$B:2 vfy-s8-new-b4=$V:4 \
    vfy-s8-base-b4=$B:4
./vfy-dump.sh vfy-p96.ids vfy-s8-dnew-b1=$V:1 vfy-s8-dnew-b3=$V:3 vfy-s8-dnew-b4=$V:4 vfy-s8-dbase-b1=$B:1
echo "session 8 done"
