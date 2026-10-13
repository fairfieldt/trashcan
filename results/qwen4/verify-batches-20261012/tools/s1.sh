#!/bin/bash
# Session 1 on invite: baseline (main 26b8d1b) against the lane-per-row kernels, per-dispatch profiles at batches 2-4,
# K-split variants, and dumps for a first correctness check.
cd ~/trashcan-exp/flashnext
ALL=TRASHCAN_VFY_Q8=rows,TRASHCAN_VFY_MOE=rows,TRASHCAN_VFY_MIX=rows
./vfy-sb.sh 64 8 8 vfy-s1-base-b3=trashcan-26b8d1b:3 vfy-s1-all-b3=trashcan-vfy:3:$ALL \
    vfy-s1-alln1-b3=trashcan-vfy:3:$ALL,TRASHCAN_VFY_N1=1 \
    vfy-s1-base-b2=trashcan-26b8d1b:2 vfy-s1-all-b2=trashcan-vfy:2:$ALL \
    vfy-s1-base-b4=trashcan-26b8d1b:4 vfy-s1-all-b4=trashcan-vfy:4:$ALL \
    vfy-s1-ks4-b3=trashcan-vfy:3:$ALL,TRASHCAN_VFY_KS=4 vfy-s1-ks8-b3=trashcan-vfy:3:$ALL,TRASHCAN_VFY_KS=8 \
    vfy-s1-ks16-b3=trashcan-vfy:3:$ALL,TRASHCAN_VFY_KS=16
./vfy-dump.sh dec-ref14.ids vfy-s1-dbase-b5=trashcan-26b8d1b:5 vfy-s1-dall-b5=trashcan-vfy:5:$ALL \
    vfy-s1-dbase-b11=trashcan-26b8d1b:11 vfy-s1-dall-b11=trashcan-vfy:11:$ALL \
    vfy-s1-dbase-b3=trashcan-26b8d1b:3 vfy-s1-dall-b3=trashcan-vfy:3:$ALL \
    vfy-s1-dbase-b1=trashcan-26b8d1b:1 vfy-s1-dalln1-b1=trashcan-vfy:1:$ALL,TRASHCAN_VFY_N1=1
echo "session 1 done"
