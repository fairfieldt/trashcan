#!/bin/bash
# Release candidate verification under one six-Mac lease (TICKETS set): warm smokes, cold-cache smokes, then greedy
# validation (candidate and base) and one decode A/B pair.
F=$HOME/dev/claude-work/trashcan-release-20261026/fallback
cd $F
scripts/smokes.sh ${WARM:-20} main/deploy/trashcan-cluster results/v5-warm
COLD=1 scripts/smokes.sh ${COLDN:-8} main/deploy/trashcan-cluster results/v5-cold
CAND=trashcan-fb5 TAG=fb5 STEPS="val14 val100" scripts/fb-val.sh
echo "verify-all done"
