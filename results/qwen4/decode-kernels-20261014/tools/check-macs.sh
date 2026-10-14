#!/bin/bash
# Binaries and kernel version on the six Macs.
# usage: check-macs.sh BINARY...
for h in incite inflight alight indite invite enlight; do
    echo "$h: $(ssh -o ConnectTimeout=5 "$h" "cd ~/trashcan-exp/flashnext && for b in $*; do sha256sum \$b | cut -c1-16; done | tr '\n' ' '; uname -r" 2>&1)"
done
