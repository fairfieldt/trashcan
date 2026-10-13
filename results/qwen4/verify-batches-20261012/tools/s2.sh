#!/bin/bash
# Session 2 on invite: main 4c1346d against the mixer and dense Q8_0 variants (per-dispatch profiles at 3-token chunks
# and single-token decode), then dumps of one combination for correctness.
cd ~/trashcan-exp/flashnext
B=trashcan-4c1346d V=trashcan-vfy
./vfy-sb.sh 64 8 8 vfy-s2-base-b3=$B:3 vfy-s2-none-b3=$V:3 \
    vfy-s2-C-b3=$V:3:TRASHCAN_VFY_DENSE=4:2:1,TRASHCAN_VFY_DOWN=4:1,TRASHCAN_VFY_UP=rows \
    vfy-s2-D-b3=$V:3:TRASHCAN_VFY_DENSE=8:4:0,TRASHCAN_VFY_DOWN=8:0,TRASHCAN_VFY_UP=rows \
    vfy-s2-E-b3=$V:3:TRASHCAN_VFY_DENSE=8:4:1,TRASHCAN_VFY_DOWN=8:1,TRASHCAN_VFY_UP=rows \
    vfy-s2-F-b3=$V:3:TRASHCAN_VFY_DENSE=2:2:1,TRASHCAN_VFY_DOWN=4:0,TRASHCAN_VFY_UP=rows \
    vfy-s2-G-b3=$V:3:TRASHCAN_VFY_DENSE=4:4:0,TRASHCAN_VFY_DOWN=4:0:n,TRASHCAN_VFY_UP=rowsn \
    vfy-s2-base-b2=$B:2 vfy-s2-C-b2=$V:2:TRASHCAN_VFY_DENSE=4:2:1,TRASHCAN_VFY_DOWN=4:1,TRASHCAN_VFY_UP=rows \
    vfy-s2-base-b4=$B:4 vfy-s2-C-b4=$V:4:TRASHCAN_VFY_DENSE=4:2:1,TRASHCAN_VFY_DOWN=4:1,TRASHCAN_VFY_UP=rows
C=TRASHCAN_VFY_DENSE=4:2:1,TRASHCAN_VFY_DOWN=4:1,TRASHCAN_VFY_UP=rows
./vfy-dump.sh dec-ref14.ids vfy-s2-dbase-b1=$B:1 vfy-s2-dC-b1=$V:1:$C vfy-s2-dbase-b3=$B:3 vfy-s2-dC-b3=$V:3:$C \
    vfy-s2-dbase-b5=$B:5 vfy-s2-dC-b5=$V:5:$C vfy-s2-dbase-b11=$B:11 vfy-s2-dC-b11=$V:11:$C
echo "session 2 done"
