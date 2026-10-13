#!/bin/bash
# One-Mac job (inflight): stage-bench of the chain binary (trashcan-pf2) against the gate/up walks unrolled only up to
# 4 slots (trashcan-pf2u4) and the DeltaNet half-row variant (trashcan-pf2hr), alternating, at 32 and 128 tokens.
cd ~/trashcan-exp/flashnext
./pf2-sb.sh "32 128" c1=trashcan-pf2 u4=trashcan-pf2u4 hr=trashcan-pf2hr
echo "pf2-s3b done"
