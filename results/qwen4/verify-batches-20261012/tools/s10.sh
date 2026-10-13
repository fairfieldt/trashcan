#!/bin/bash
# Session 10 on invite (one-Mac lease, ~8 min): the padding changes left out of the final tree. trashcan-vfy11a: Q4_0
# and IQ4_XS repacks as on main, arena buffers on 64-byte boundaries; trashcan-vfy11b: no alignment change at all;
# against trashcan-vfy10 (both): 3-token chunks with single-token decode, alternated twice; 96-token dumps.
cd ~/trashcan-exp/flashnext || exit 1
X=trashcan-vfy10 A=trashcan-vfy11a N=trashcan-vfy11b
./vfy-sb.sh 64 16 8 vfy-s10-x-b3=$X:3 vfy-s10-n-b3=$N:3 vfy-s10-a-b3=$A:3 vfy-s10-x2-b3=$X:3 vfy-s10-n2-b3=$N:3 \
    vfy-s10-a2-b3=$A:3
./vfy-dump.sh vfy-p96.ids vfy-s10-dn-b1=$N:1 vfy-s10-dn-b4=$N:4 vfy-s10-da-b1=$A:1
echo "session 10 done"
