#!/bin/bash
# Session 6 on invite (one-Mac lease, ~9 min): experiments in trashcan-vfy7 (this branch's kernels plus switches): the
# mixer's first half with 2 rows per wave (DOWN=r2) and/or its finishing pass as a separate dispatch (r4n, r2n), the
# out projections with K split over two waves for one token (OUT=k2); X0 = the same binary without switches.
cd ~/trashcan-exp/flashnext
X=trashcan-vfy7
A=TRASHCAN_VFY_DOWN=r2,TRASHCAN_VFY_OUT=k2 Bn=TRASHCAN_VFY_DOWN=r4n Cn=TRASHCAN_VFY_DOWN=r2n
./vfy-sb.sh 64 16 8 vfy-s6-X0-b3=$X:3 vfy-s6-A-b3=$X:3:$A vfy-s6-B-b3=$X:3:$Bn vfy-s6-C-b3=$X:3:$Cn \
    vfy-s6-X0-b2=$X:2 vfy-s6-A-b2=$X:2:$A vfy-s6-C-b2=$X:2:$Cn vfy-s6-X0-b4=$X:4 vfy-s6-A-b4=$X:4:$A vfy-s6-C-b4=$X:4:$Cn
./vfy-dump.sh vfy-p96.ids vfy-s6-dA-b1=$X:1:$A vfy-s6-dC-b3=$X:3:$Cn
echo "session 6 done"
