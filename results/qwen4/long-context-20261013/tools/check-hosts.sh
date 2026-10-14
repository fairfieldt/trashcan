#!/bin/bash
# Before a session: the binaries (md5), scripts, prompts and model on every Mac. No GPU use.
# usage: check-hosts.sh BINARY...
W=trashcan-exp/flashnext
for h in ${HOSTS:-incite indite inflight alight invite enlight}; do
    echo "== $h"
    ssh "$h" "cd $W && md5sum $* 2>&1 | cut -c1-90; ls long-*.sh long.ids long-*.ids scripts/latency-run.py scripts/with-clocks.sh 2>&1 | tr '\n' ' '; echo; ls -d /home/fairfieldt/models/qwen3.8-flash-next/G-iq4 2>&1"
done
