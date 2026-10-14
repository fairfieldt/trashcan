#!/bin/bash
# Session 4's single-Mac part: six single-Mac jobs (one per Mac), queued at once; the coordinator's batcher runs them
# between six-Mac sessions, before long-s4. NEW = trashcan-long-r19 (this round on main 4307169), BASE =
# trashcan-4307169 (main); r17/r18 (on main 7da6f93) differ only in the decode attention and its merge (r18), r19x is
# r19 without qwen4_matf32_t. Logs to s4-1mac-MACHINE.log here.
#  incite, indite: stage-bench profiles at 8320/30080, new and base (opposite orders); indite also r17 and r18 at 8320;
#  invite: stage-bench profiles of new and base at 128, 1024, 2176 and 16512, r19x at 128;
#  enlight: one QSA layer at 204,800 positions with a Q8 K/V cache, new and base;
#  alight: new against base, dumps of the last token at 8196 and 30000; r17 against r18 with a Q8 K/V cache at 8196;
#  inflight: r17 against r18 at 8196 and 30000 (the same bits expected); block scores with the leaves as a fused fma
#  (r15f) against the replaced kernel's bits.
L=$HOME/dev/claude-work/flashnext-20261012/long
NEW=trashcan-long-r19 BASE=trashcan-4307169 R17=trashcan-long-r17 R18=trashcan-long-r18 X=trashcan-long-r19x
j() { local m=$1 est=$2; shift 2; "$L/long-1mac.sh" "$m" "long-s4-$m" "$est" -- "$*" > "$L/s4-1mac-$m.log" 2>&1 & }
j incite 8 "./long-sb.sh $NEW s4n-a 8320 30080; ./long-sb.sh $BASE s4b-a 8320 30080"
j indite 10 "./long-sb.sh $BASE s4b-b 8320 30080; ./long-sb.sh $NEW s4n-b 8320 30080; ./long-sb.sh $R17 s4o-b 8320; ./long-sb.sh $R18 s4r-b 8320"
j invite 8 "./long-sb.sh $NEW s4n-c 128 1024 2176 16512; ./long-sb.sh $BASE s4b-c 128 1024 2176 16512; ./long-sb.sh $X s4x-c 128"
j enlight 9 "./long-200k.sh $NEW long-sb-s4n-200k; ./long-200k.sh $BASE long-sb-s4b-200k"
j alight 8 "./long-equiv.sh $BASE $NEW s4bn 8196 30000; KV=q8 ./long-equiv.sh $R17 $R18 s4oq8 8196"
j inflight 10 "./long-equiv.sh $R17 $R18 s4on 8196 30000; ./long-unit.sh trashcan-long-r15f s4-fma 8196 30000"
wait
echo "s4-1mac done"
