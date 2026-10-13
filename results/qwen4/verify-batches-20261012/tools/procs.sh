#!/bin/bash
# List engine processes of one binary name on all six Macs (read only).
# usage: procs.sh BINARY
for h in incite indite inflight alight invite enlight; do
    echo "$h: $(ssh -- "$h" "pgrep -fa 'trashcan-exp/flashnext/$1 qwen4|\\./$1 qwen4' | grep -v pgrep | cut -c1-90" | tr '\n' ' ')"
done
