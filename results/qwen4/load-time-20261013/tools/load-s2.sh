#!/bin/bash
# Load-time session 2 on indite (under one gpuq lease), after the rebase onto main 7da6f93: weight hashes against
# 7da6f93, stage-bench with several --batch sizes on one load against separate runs, trashcan-load against
# trashcan-7da6f93 warm and cold, alternating, the timing build, and the p100 dumps.
cd ~/trashcan-exp/flashnext || exit 1
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
P="--tokens-file /home/fairfieldt/trashcan-exp/flashnext/dec-p100.ids"
./load-sb.sh hashmain7da-w1 trashcan-load-hash-main7da warm
./load-sb.sh hashnew7da-w1 trashcan-load-hash-new7da warm
diff <(grep "^HASH" results/load-hashmain7da-w1.err) <(grep "^HASH" results/load-hashnew7da-w1.err) > /dev/null \
    && echo "weight hashes: $(grep -c '^HASH' results/load-hashnew7da-w1.err) buffers, all identical" || echo "weight hashes DIFFER"
A="--ranges 0:4,4:8 --gpus 0,1 --context 32768 --head-rows 0:21504,21504:43008 --prefill 3 --decode 8 --start 64"
t() { local s=$(date +%s.%N); "$@"; echo "$(echo "$(date +%s.%N) - $s" | bc) s"; }
echo "multi 16,32,64,128: $(t ./trashcan-load qwen4 stage-bench $M $A --batch 16,32,64,128 > results/load-multi.json 2> results/load-multi.err)"
for b in 16 32 64 128; do
    echo "single $b: $(t ./trashcan-load qwen4 stage-bench $M $A --batch $b > results/load-single-$b.json 2> results/load-single-$b.err)"
done
echo "main single 128: $(t ./trashcan-7da6f93 qwen4 stage-bench $M $A --batch 128 > results/load-main-single-128.json 2> results/load-main-single-128.err)"
./load-sb.sh base7da-w1 trashcan-7da6f93 warm
./load-sb.sh new-w1 trashcan-load warm
./load-sb.sh base7da-w2 trashcan-7da6f93 warm
./load-sb.sh new-w2 trashcan-load warm
./load-sb.sh prof4-w1 trashcan-load-prof4 warm
./load-sb.sh base7da-c1 trashcan-7da6f93 cold
./load-sb.sh new-c1 trashcan-load cold
./load-sb.sh base7da-c2 trashcan-7da6f93 cold
./load-sb.sh new-c2 trashcan-load cold
./load-sb.sh prof4-c1 trashcan-load-prof4 cold
./load-sb.sh dump-main7da trashcan-7da6f93 warm $P --out results/load-dump-main7da
./load-sb.sh dump-new trashcan-load warm $P --out results/load-dump-new
n=0; d=0
for f in results/load-dump-main7da/*.f32; do
    n=$((n + 1)); cmp -s $f results/load-dump-new/$(basename $f) || { d=$((d + 1)); echo "differs: $f"; }
done
echo "dump new against main 7da6f93: $n tensors, $d differ; index $(cmp -s results/load-dump-main7da/index.tsv results/load-dump-new/index.tsv && echo identical || echo DIFFERENT)"
echo "session 2 done"
