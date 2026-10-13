#!/bin/bash
# Session 3 on enlight (under a lease): prefetch variants, push-constant batch, top-k variants, the prefill router.
cd ~/trashcan-exp/flashnext
B=trashcan-4c1346d N=trashcan-moe
./moe-sb.sh prof moe-s3-new-b1 $N 1
./moe-sb.sh prof moe-s3-g3n-b1 $N 1 TRASHCAN_MOE_AB=g3n
./moe-sb.sh prof moe-s3-g3w-b1 $N 1 TRASHCAN_MOE_AB=g3w
./moe-sb.sh prof moe-s3-d3w-b1 $N 1 TRASHCAN_MOE_AB=d3w
./moe-sb.sh prof moe-s3-nor3-b1 $N 1 TRASHCAN_MOE_AB=nor3
./moe-sb.sh prof moe-s3-tka-b1 $N 1 TRASHCAN_MOE_AB=tka
./moe-sb.sh prof moe-s3-fused-b1 $N 1 TRASHCAN_MOE_AB=fused
./moe-sb.sh prof moe-s3-base-b1 $B 1
./moe-sb.sh prof moe-s3-new-b3 $N 3
./moe-sb.sh prof moe-s3-g3w-b3 $N 3 TRASHCAN_MOE_AB=g3w
./moe-sb.sh prof moe-s3-fused-b3 $N 3 TRASHCAN_MOE_AB=fused
./moe-sb.sh prof moe-s3-new-b4 $N 4
./moe-sb.sh prof moe-s3-base-b4 $B 4
./moe-sb.sh prof moe-s3-new-b2 $N 2
./moe-sb.sh prof moe-s3-base-b2 $B 2
./moe-sb.sh prof moe-s3-new-b16 $N 16
./moe-sb.sh prof moe-s3-base-b16 $B 16
./moe-sb.sh prof moe-s3-new-b128 $N 128
./moe-sb.sh prof moe-s3-base-b128 $B 128
./moe-sb.sh dump moe-s3-dump-new-b1 $N 1 dec-ref14.ids
./moe-sb.sh dump moe-s3-dump-new-p16 $N 16 moe-p100.ids
./moe-sb.sh dump moe-s3-dump-base-p16 $B 16 moe-p100.ids
./moe-sb.sh dump moe-s3-dump-fused-b3 $N 3 dec-ref14.ids TRASHCAN_MOE_AB=fused
echo "moe-s3 done"
