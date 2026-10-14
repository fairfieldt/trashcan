#!/bin/bash
# Copy the kernel-check scripts and test prompts to ~/trashcan-exp/flashnext on every Mac.
L=$HOME/dev/claude-work/flashnext-20261012/long
for h in ${HOSTS:-incite indite inflight alight invite enlight}; do
    rsync -a "$L/long-unit.sh" "$L/long-sb.sh" "$L/long-equiv.sh" "$L/long-200k.sh" "$L/long-chk3061.sh" "$L"/long-2052.ids "$L"/long-2101.ids \
        "$L"/long-8193.ids "$L"/long-8196.ids "$L"/long-30000.ids "$L"/long-30081.ids "${h}:trashcan-exp/flashnext/" \
        && echo "$h ok" &
done
wait
