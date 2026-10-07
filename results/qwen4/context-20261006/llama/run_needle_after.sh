#!/bin/bash
# Start the needle tests once the CPU quality runs are done (they cannot share the Mac's memory with Metal).
Q=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash
until grep -q "q2dn quality done" "$Q/q2dn-quality.out" 2>/dev/null; do sleep 15; done
cd "$Q/longctx" || exit 1
python3 needle.py 100000 200000
echo "needle done"
