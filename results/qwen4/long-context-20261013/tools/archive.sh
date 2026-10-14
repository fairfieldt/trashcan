#!/bin/bash
# Copy the round's measurements and check outputs into results/qwen4/long-context-20261013 (no GPU use): chain runs of
# long-s1 (baseline timeline) and long-s5 (driver JSON/err, node logs, clocks), single-Mac stage-bench JSON and clock
# logs, session logs, and the comparison outputs as text (the f32 dumps stay out: 3.6 GB).
L=$HOME/dev/claude-work/flashnext-20261012/long
R=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a43eb2477e990f76f/results/qwen4/long-context-20261013
cd "$L" || exit 1
mkdir -p "$R/runs" "$R/sb" "$R/logs"
cp -R runs/long-s5-* runs/long-s1-* "$R/runs/"
cp sb/long-sb-s4* sb/long-sb-base-* sb/long-sb-s3-* "$R/sb/"
cp s5.log sv.log chk-alight2.log s3.log s4-1mac-*.log s4b-1mac-*.log s4c-1mac-*.log s4d-1mac-incite.log "$R/logs/" 2>/dev/null
./chk-compare.sh alight c2 > "$R/logs/chk3061-alight-compare.txt" 2>&1
./s4-1mac-check.sh 2>/dev/null | grep -v '^sb/\|^  \|Traceback\|json.decoder' > "$R/logs/single-mac-dump-checks.txt"
./s4b-check.sh 2>/dev/null | head -12 > "$R/logs/r20-vs-r19-dumps.txt"
python3 boundscheck.py 3061 64 --decode 10 > "$R/logs/boundscheck-3061.txt"
python3 clocks.py runs/long-s5-* > "$R/logs/s5-clocks.txt"
cp clocks.py "$R/tools/"
du -sh "$R"
