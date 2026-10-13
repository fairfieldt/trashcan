#!/bin/bash
# Session 4 on invite: the cleaned-up kernels (trashcan-vfy2: exact-column Q8_0 kernels, qwen4_mix_down_skw,
# qwen4_mix_rows_up) against main 4c1346d: stage-bench at batches 2-4 (with single-token decode), alternated, and
# dumps of the 96-token prompt in chunks of 1, 2, 3 and 4 tokens from both binaries. trashcan-vfy3: qwen4_mix_rows_up
# loading its epilogue inputs in the first wave only, and the experiment qwen4_mix_down_skl (TRASHCAN_VFY_SKL=1).
cd ~/trashcan-exp/flashnext
B=trashcan-4c1346d V=trashcan-vfy2 X=trashcan-vfy3 S=TRASHCAN_VFY_SKL=1
./vfy-sb.sh 64 16 8 vfy-s4-base-b3=$B:3 vfy-s4-new-b3=$V:3 vfy-s4-w0-b3=$X:3 vfy-s4-skl-b3=$X:3:$S \
    vfy-s4-base-b2=$B:2 vfy-s4-new-b2=$V:2 vfy-s4-skl-b2=$X:2:$S vfy-s4-base-b4=$B:4 vfy-s4-new-b4=$V:4 \
    vfy-s4-skl-b4=$X:4:$S vfy-s4-base2-b3=$B:3 vfy-s4-new2-b3=$V:3 vfy-s4-w02-b3=$X:3 vfy-s4-skl2-b3=$X:3:$S
./vfy-dump.sh vfy-p96.ids vfy-s4-dbase-b1=$B:1 vfy-s4-dnew-b1=$V:1 vfy-s4-dbase-b2=$B:2 vfy-s4-dnew-b2=$V:2 \
    vfy-s4-dbase-b3=$B:3 vfy-s4-dnew-b3=$V:3 vfy-s4-dbase-b4=$B:4 vfy-s4-dnew-b4=$V:4 vfy-s4-dskl-b1=$X:1:$S \
    vfy-s4-dskl-b3=$X:3:$S vfy-s4-dw0-b1=$X:1
echo "session 4 done"
