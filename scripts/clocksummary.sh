#!/bin/bash
# Summarizes a clockmon.sh log: how many samples each card spent at each sclk/mclk pair, and max temperature.
# usage: clocksummary.sh CLOCKS.txt
awk 'NR > 1 {
        c0[$2 "/" $3]++; c1[$5 "/" $6]++
        if ($4 > t0) t0 = $4
        if ($7 > t1) t1 = $7
        n++
     }
     END {
        printf "%d samples\n", n
        printf "card0 (02:00.0) sclk/mclk:"; for (k in c0) printf " %s x%d", k, c0[k]; printf "  max %dC\n", t0
        printf "card1 (06:00.0) sclk/mclk:"; for (k in c1) printf " %s x%d", k, c1[k]; printf "  max %dC\n", t1
     }' "$1"
