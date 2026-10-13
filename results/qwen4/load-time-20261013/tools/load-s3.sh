#!/bin/bash
# Load-time session 3 on indite (under one gpuq lease): the final branch (in-flight upload guard added): weight hashes
# against 7da6f93 (results/load-hashmain7da-w1.err from session 2), warm and cold loads; and which out-projection
# kernels a 16-token prefill chunk runs with --batch 16,128 (one load) and --batch 16 alone (TRASHCAN_Q4_PROFILE=1).
cd ~/trashcan-exp/flashnext || exit 1
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
./load-sb.sh hashfinal-w1 trashcan-load-hash-final warm
diff <(grep "^HASH" results/load-hashmain7da-w1.err) <(grep "^HASH" results/load-hashfinal-w1.err) > /dev/null \
    && echo "weight hashes: $(grep -c '^HASH' results/load-hashfinal-w1.err) buffers, all identical to 7da6f93" \
    || echo "weight hashes DIFFER"
./load-sb.sh final-w1 trashcan-load warm
./load-sb.sh final-w2 trashcan-load warm
./load-sb.sh final-c1 trashcan-load cold
A="--ranges 0:4 --gpus 0 --context 32768 --prefill 1 --decode 1 --start 16"
TRASHCAN_Q4_PROFILE=1 ./trashcan-load qwen4 stage-bench $M $A --batch 16,128 > results/load-outproj-multi.json 2> results/load-outproj-multi.err
TRASHCAN_Q4_PROFILE=1 ./trashcan-load qwen4 stage-bench $M $A --batch 16 > results/load-outproj-16.json 2> results/load-outproj-16.err
python3 load-outproj.py results/load-outproj-multi.json results/load-outproj-16.json
echo "session 3 done"
