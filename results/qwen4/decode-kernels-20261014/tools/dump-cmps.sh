#!/bin/bash
# Every stage-dump comparison of the round against main's dumps, one file.
dk=/Users/fairfieldt/dev/claude-work/flashnext-20261012/dk2
c="python3 $dk/tools/cmpdump.py"
cd "$dk" || exit 1
echo "# s1: lite barriers (TRASHCAN_Q4_DIAG=lite), 14 tokens one at a time"; $c s1/dump-lite s1/dump-main 3
echo "# s2: coherent bindings + execution-only barriers"
for d in b1 b4 p16; do $c s2/dump-cohx-$d s2/dump-main-$d 3; done
echo "# s3: v2 kernels"; for d in b1 b4; do $c s3/dump-v2-$d s3/dump-main-$d 3; done
echo "# s4: gate/up 2 + 2 rows (v3), fused mixer (v3f)"
$c s4/dump-v3-b1 s4/dump-main-b1 3; $c s4/dump-v3f-b1 s4/dump-main-b1 3; $c s4/dump-v3f-p16 s4/dump-main-p16 3
echo "# s6: mixer variants k5 + upl"; $c s6/dump-both-b1 s6/dump-main-b1 3; $c s6/dump-upl-b4 s6/dump-main-b4 3
echo "# s7: v5 (c1 + attention queries via LDS + gdn unconditional load)"
for d in b1 b4 p16 l2101; do $c s7/dump-v5-$d s7/dump-main-$d 3; done
if [ -d s8 ]; then
    echo "# s8: c1 with Q8 K/V caches"
    for b in 1 16; do $c s8/dump-q8-trashcan-dk2-c1-b$b s8/dump-q8-trashcan-15e5457-b$b 3; done
fi
echo "# s10: c2 (light barriers) against main 9baedca"; for d in b1 b4 p16 l2101; do $c s10/dump-c2-$d s10/dump-main-$d 3; done
echo "# s11: c3 (c2 + qwen4_moe_down_p, removed) against main 9baedca"
for d in b1 l2101; do $c s11/dump-c3-$d s10/dump-main-$d 3; done
echo "# stress (c1 against main 15e5457, layers 0-7 on both GPUs of indite)"; bash tools/stress-cmp.sh c1-stress
echo "# stress (c2 against main 9baedca, layers 0-7 on both GPUs of enlight)"; bash tools/stress-cmp.sh c2-stress
