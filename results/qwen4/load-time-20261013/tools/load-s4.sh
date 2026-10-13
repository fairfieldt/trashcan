#!/bin/bash
# Load-time session 4 on indite (under one gpuq lease): the branch rebased onto main 4307169 (trashcan-load-4307):
# weight hashes and the p100 dump against trashcan-4307169, and one alternated pair of warm loads.
cd ~/trashcan-exp/flashnext || exit 1
P="--tokens-file /home/fairfieldt/trashcan-exp/flashnext/dec-p100.ids"
./load-sb.sh hashmain4307-w1 trashcan-load-hash-main4307 warm
./load-sb.sh hashnew4307-w1 trashcan-load-hash-new4307 warm
diff <(grep "^HASH" results/load-hashmain4307-w1.err) <(grep "^HASH" results/load-hashnew4307-w1.err) > /dev/null \
    && echo "weight hashes: $(grep -c '^HASH' results/load-hashnew4307-w1.err) buffers, all identical to 4307169" \
    || echo "weight hashes DIFFER"
./load-sb.sh base4307-w1 trashcan-4307169 warm
./load-sb.sh new4307-w1 trashcan-load-4307 warm
./load-sb.sh dump-main4307 trashcan-4307169 warm $P --out results/load-dump-main4307
./load-sb.sh dump-new4307 trashcan-load-4307 warm $P --out results/load-dump-new4307
n=0; d=0
for f in results/load-dump-main4307/*.f32; do
    n=$((n + 1)); cmp -s $f results/load-dump-new4307/$(basename $f) || { d=$((d + 1)); echo "differs: $f"; }
done
echo "dump rebased against main 4307169: $n tensors, $d differ; index $(cmp -s results/load-dump-main4307/index.tsv results/load-dump-new4307/index.tsv && echo identical || echo DIFFERENT)"
echo "session 4 done"
