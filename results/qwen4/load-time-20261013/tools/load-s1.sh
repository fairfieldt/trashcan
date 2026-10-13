#!/bin/bash
# Load-time session 1 on indite (under one gpuq lease): weight hashes main vs new, warm A/B, timing builds warm and
# cold, stage-bench dumps for the p100 prompt.
cd ~/trashcan-exp/flashnext || exit 1
P="--tokens-file /home/fairfieldt/trashcan-exp/flashnext/dec-p100.ids"
./load-sb.sh hashmain-w1 trashcan-load-hash-main warm
./load-sb.sh hashnew-w1 trashcan-load-hash-new warm
./load-sb.sh base-w3 trashcan-6f6635a warm
./load-sb.sh v1-w1 trashcan-load-v1 warm
./load-sb.sh base-w4 trashcan-6f6635a warm
./load-sb.sh v1-w2 trashcan-load-v1 warm
./load-sb.sh prof1-w1 trashcan-load-prof1 warm
./load-sb.sh prof1-c1 trashcan-load-prof1 cold
./load-sb.sh v1-c1 trashcan-load-v1 cold
./load-sb.sh v2-w1 trashcan-load-v2 warm
./load-sb.sh prof2-w1 trashcan-load-prof2 warm
./load-sb.sh prof2-c1 trashcan-load-prof2 cold
./load-sb.sh v2-c1 trashcan-load-v2 cold
./load-sb.sh v1-c2 trashcan-load-v1 cold
./load-sb.sh dump-main trashcan-6f6635a warm $P --out results/load-dump-main
./load-sb.sh dump-v1 trashcan-load-v1 warm $P --out results/load-dump-v1
S5=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00005-of-00006.gguf
for bs in 64M 4M 1M 128K; do echo "dd direct bs=$bs: $(dd if=$S5 of=/dev/null bs=$bs iflag=direct,skip_bytes skip=$((8 << 30)) count=$(( (2 << 30) / $(numfmt --from=iec $bs) )) 2>&1 | tail -1)"; done
echo "session 1 done"
