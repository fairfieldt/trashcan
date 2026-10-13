#!/bin/bash
# Session 7 on invite (one-Mac lease, ~8 min): the out projections (2560 x 6144) with each row's K split over two waves
# (qwen4_q8_mvk2/_mvk4) against qwen4_q8_mvw4 in trashcan-vfy8 (this branch plus the switch): OUT=k2 (2 rows per wave
# for one token, 4 for more), OUT=k22 (2 rows always), X = no switch; dumps for batches against single tokens.
cd ~/trashcan-exp/flashnext
X=trashcan-vfy8 K=TRASHCAN_VFY_OUT=k2 KK=TRASHCAN_VFY_OUT=k22
./vfy-sb.sh 64 16 8 vfy-s7-X-b3=$X:3 vfy-s7-k2-b3=$X:3:$K vfy-s7-k22-b3=$X:3:$KK vfy-s7-X-b2=$X:2 vfy-s7-k2-b2=$X:2:$K \
    vfy-s7-k22-b2=$X:2:$KK vfy-s7-X-b4=$X:4 vfy-s7-k2-b4=$X:4:$K vfy-s7-k22-b4=$X:4:$KK
./vfy-dump.sh vfy-p96.ids vfy-s7-dk2-b1=$X:1:$K vfy-s7-dk2-b2=$X:2:$K vfy-s7-dk2-b4=$X:4:$K
echo "session 7 done"
