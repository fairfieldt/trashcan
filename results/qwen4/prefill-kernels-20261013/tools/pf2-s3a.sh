#!/bin/bash
# One-Mac job (alight): stage-bench of the chain binary (trashcan-pf2, 583d2dc) against the expert walks with the line's
# blocks unrolled (trashcan-pf2u), alternating, at every chunk size.
cd ~/trashcan-exp/flashnext
./pf2-sb.sh "16 32 64 128" c1=trashcan-pf2 u=trashcan-pf2u
echo "pf2-s3a done"
