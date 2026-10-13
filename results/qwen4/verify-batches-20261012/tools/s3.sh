#!/bin/bash
# Session 3 on invite: correctness dumps of each variant on its own (session 2's combination C was wrong), then
# unrolled single-token dense kernels, mixer first-half chunking and load widths, whole-row second half; batches of
# 3 (with single-token decode), 2 and 4.
cd ~/trashcan-exp/flashnext
B=trashcan-4c1346d V=trashcan-vfy
./vfy-dump.sh dec-ref14.ids vfy-s3-d-down0-b1=$V:1:TRASHCAN_VFY_DOWN=4:0 vfy-s3-d-down1-b1=$V:1:TRASHCAN_VFY_DOWN=4:1 \
    vfy-s3-d-up-b1=$V:1:TRASHCAN_VFY_UP=rows vfy-s3-d-upf-b1=$V:1:TRASHCAN_VFY_UP=rowsf \
    vfy-s3-d-dense1-b1=$V:1:TRASHCAN_VFY_DENSE=4:2:1 vfy-s3-d-denseu-b1=$V:1:TRASHCAN_VFY_DENSE=2:2:u \
    vfy-s3-d-cb16-b1=$V:1:TRASHCAN_VFY_DOWN=4:cb16 vfy-s3-d-w80-b1=$V:1:TRASHCAN_VFY_DOWN=4:w80 \
    vfy-s3-d-w16-b1=$V:1:TRASHCAN_VFY_DOWN=4:w16 vfy-s3-d-dense0-b3=$V:3:TRASHCAN_VFY_DENSE=4:4:0 \
    vfy-s3-d-down0-b5=$V:5:TRASHCAN_VFY_DOWN=4:0 vfy-s3-d-up-b5=$V:5:TRASHCAN_VFY_UP=rows
c2=TRASHCAN_VFY_DENSE=1:1:u,TRASHCAN_VFY_DOWN=4:cb16,TRASHCAN_VFY_UP=rows
c3=TRASHCAN_VFY_DENSE=2:2:u,TRASHCAN_VFY_DOWN=4:w80,TRASHCAN_VFY_UP=rowsf
c4=TRASHCAN_VFY_DENSE=4:8:u,TRASHCAN_VFY_DOWN=4:w16,TRASHCAN_VFY_UP=rows
c5=TRASHCAN_VFY_DENSE=4:4:0,TRASHCAN_VFY_DOWN=4:0,TRASHCAN_VFY_UP=rowsf
./vfy-sb.sh 64 8 8 vfy-s3-base-b3=$B:3 vfy-s3-c2-b3=$V:3:$c2 vfy-s3-c3-b3=$V:3:$c3 vfy-s3-c4-b3=$V:3:$c4 \
    vfy-s3-c5-b3=$V:3:$c5 vfy-s3-base-b2=$B:2 vfy-s3-c2-b2=$V:2:$c2 vfy-s3-c3-b2=$V:2:$c3 vfy-s3-c4-b2=$V:2:$c4 \
    vfy-s3-c5-b2=$V:2:$c5 vfy-s3-base-b4=$B:4 vfy-s3-c2-b4=$V:4:$c2 vfy-s3-c3-b4=$V:4:$c3 vfy-s3-c4-b4=$V:4:$c4 \
    vfy-s3-c5-b4=$V:4:$c5
echo "session 3 done"
