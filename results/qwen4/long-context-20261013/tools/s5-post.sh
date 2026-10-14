#!/bin/bash
# After long-s5 (no GPU use): fetch every run (driver JSON/err, node logs, clocks) into runs/, summarize, and analyse
# the timeline run.
L=$HOME/dev/claude-work/flashnext-20261012/long
cd "$L" || exit 1
"$L/fetch.sh" $(cat s5-names.txt) > /dev/null
python3 s4sum.py s5
tl=runs/long-s5-tl-pp8192-new
[ -e $tl/long-s5-tl-pp8192-new.err ] && python3 timeline.py $tl/long-s5-tl-pp8192-new.err $tl/node-long-s5-tl-pp8192-new-*.log
