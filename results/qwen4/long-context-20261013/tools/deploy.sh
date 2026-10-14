#!/bin/bash
# Copy a cross-built binary to ~/trashcan-exp/flashnext/NAME on the six Macs (or HOSTS).
# usage: deploy.sh BINARY NAME
set -euo pipefail
bin=$1 name=$2
md5 -q "$bin"
for h in ${HOSTS:-incite indite inflight alight invite enlight}; do
    rsync -a "$bin" "${h}:trashcan-exp/flashnext/${name}" && echo "$h ok" &
done
wait
