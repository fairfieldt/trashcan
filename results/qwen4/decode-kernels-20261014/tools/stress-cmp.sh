#!/bin/bash
# Compare the stress check's dumps: candidate against main for every prompt length and repetition, and each binary's
# two repetitions against each other.
# usage: stress-cmp.sh DIR
d=$1
here=$(dirname "$0")
for r in 1 2; do
    for n in 520 1030 2060 2600 2600b4; do
        python3 "$here/cmpdump.py" "$d/cand-n$n-r$r" "$d/main-n$n-r$r" 3 | head -2
    done
done
for b in main cand; do
    for n in 520 1030 2060 2600 2600b4; do
        python3 "$here/cmpdump.py" "$d/$b-n$n-r2" "$d/$b-n$n-r1" 3 | head -1
    done
done
