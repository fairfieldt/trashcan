#!/bin/bash
# Copy main's binary and the candidate (built with scripts/cross-build.sh, kept in ~/dev/claude-work/mixcopies-20261023)
# to ~/trashcan-exp/flashnext on every Mac, and the session script to incite.
set -euo pipefail
B=/Users/fairfieldt/dev/claude-work/mixcopies-20261023
for h in incite indite inflight alight invite enlight; do
    rsync -a $B/trashcan-b7f4979 $B/trashcan-mixc1 "$h:trashcan-exp/flashnext/" && echo "$h ok"
done
rsync -a "$(dirname "$0")/mc-s1.sh" incite:trashcan-exp/flashnext/
