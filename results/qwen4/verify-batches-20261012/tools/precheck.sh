#!/bin/bash
# Check that the binaries and inputs of a six-Mac session exist on every Mac (and the driver's inputs on incite).
# usage: precheck.sh BINARY...
set -uo pipefail
ok=1
for h in incite indite inflight alight invite enlight; do
    for b in "$@"; do
        ssh "$h" "test -x ~/trashcan-exp/flashnext/$b" || { echo "$h: missing $b"; ok=0; }
    done
    ssh "$h" "test -f /home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf" || { echo "$h: missing model"; ok=0; }
done
for f in /home/fairfieldt/trashcan-exp/flashnext/long.ids /home/fairfieldt/trashcan-exp/flashnext/p100.ids \
    /home/fairfieldt/trashcan-exp/flashnext/mtp-prompts/water.ids /home/fairfieldt/trashcan-exp/flashnext/mtp-prompts/csv.ids \
    /home/fairfieldt/trashcan-exp/flashnext/mtp-prompts/transformer.ids \
    /home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf; do
    ssh incite "test -f $f" || { echo "incite: missing $f"; ok=0; }
done
[ $ok = 1 ] && echo "precheck ok: $* on all six Macs, inputs on incite"
