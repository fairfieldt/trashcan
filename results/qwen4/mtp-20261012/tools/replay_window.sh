#!/bin/bash
# Tokens per round (k = 3, no rewrite, 64K head, 4-bit dense) with the block fed only the last prefill chunk's rows
# under main's adaptive chunk sizes (water 7, csv 7, transformer 9, long 53) against the whole prompt / 64 rows.
cd /Users/fairfieldt/dev/claude-work/flashnext-20261012/mtp || exit 1
mkdir -p accept/window
T=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-ab5cf8f185c84b7fe/target/release/trashcan
G=/Volumes/Green/Qwen/flash-next/cand/G-iq4/G-iq4-00001-of-00006.gguf
M=/Volumes/Green/Qwen/flash-next/mtp/mtp-G-iq4.gguf
run() { $T qwen4 mtp replay $G $M accept/run2/$1.json accept/window/$1-w$2.json --vocab 65536 --depth 3 --rounds --window $2 --threads 8 > accept/window/$1-w$2.log 2>&1; echo "$1 w$2 rc $?"; }
run water 7; run water 128; run csv 7; run csv 128; run transformer 9; run transformer 128; run long 53; run long 64
python3 - <<'PY'
import json
for p, a, b in [('water', 7, 128), ('csv', 7, 128), ('transformer', 9, 128), ('long', 53, 64)]:
    r = [json.load(open(f'accept/window/{p}-w{w}.json')) for w in (a, b)]
    print(p, *[f"w{w}: {x['tokens'] / x['rounds']:.3f}" for w, x in zip((a, b), r)])
for ws in [(7, 7, 9, 53), (128, 128, 128, 64)]:
    t = r = 0
    for p, w in zip(['water', 'csv', 'transformer', 'long'], ws):
        x = json.load(open(f'accept/window/{p}-w{w}.json')); t += x['tokens']; r += x['rounds']
    print(ws, f'pooled {t / r:.3f}')
PY
