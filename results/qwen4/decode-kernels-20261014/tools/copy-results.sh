#!/bin/bash
# Copy the single-Mac sessions' JSON, stderr and clock logs (not the stage dumps) into the results directory, by Mac,
# and the round's tools.
set -uo pipefail
dk=/Users/fairfieldt/dev/claude-work/flashnext-20261012/dk2
R=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a70283bc249c19df0/results/qwen4/decode-kernels-20261014
copy() { # host session
    [ -d "$dk/$2" ] || return
    mkdir -p "$R/$1/$2"
    find "$dk/$2" -maxdepth 1 -type f \( -name '*.json' -o -name '*.err' -o -name '*.clk' -o -name '*.txt' \) \
        -exec cp {} "$R/$1/$2/" \;
}
for s in s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 c1-stress; do copy indite $s; done
copy enlight c2-stress
# The six-Mac sessions, without the validation dumps (bit-identical to main's in results/qwen4/main-suite-20261013:
# validation.txt in each session).
mkdir -p "$R/chain"
for t in dk2c1 dk2c2 dk2c2r; do
    rsync -a --exclude "$t-val14/" --exclude "$t-val100/" "$dk/chain/$t/" "$R/chain/$t/"
done
cp "$dk/dump-cmps.txt" "$R/dumps-compared.txt"
mkdir -p "$R/tools"
for t in "$dk"/tools/*; do
    case $(basename "$t") in
        dk2-stress-c2.sh|deploy-bin.sh|fetch-clocks.sh|queues.sh|kill-nodes.sh) ;;  # this session's helpers
        *) cp "$t" "$R/tools/" ;;
    esac
done
du -sh "$R/indite" "$R/enlight" "$R/chain"
