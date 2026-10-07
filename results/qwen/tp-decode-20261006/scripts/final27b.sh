#!/bin/bash
# Final 27B TP4 runs from the Mac: one node (final binary, --mtp) on invite, then benches and correctness on incite.
# usage: final27b.sh INCITE_LEASE INVITE_LEASE
li=$1; lv=$2
M='~/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf'
MTP='~/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf'
S=/dev/tbstream0,/dev/tbstream1,/dev/tbstream2
ssh invite "pkill -f '^[.]/trashcan(-base)? qwen-tp-node'; sleep 2; cd ~/trashcan-exp/decode && gpuq run $lv -- ./node.sh node-final-27b ./trashcan $M --mtp $MTP --stream $S" > /dev/null
sleep 2
ssh incite "cd ~/trashcan-exp/decode && gpuq run $li -- bash -c './tp4.sh final-27b-a ./trashcan; ./tp4.sh final-27b-b ./trashcan; ./tp4.sh final-27b-mtp-a ./trashcan --mtp $MTP; ./tp4.sh final-27b-mtp-b ./trashcan --mtp $MTP; ./tp4check.sh val-27b-final ./trashcan validate; ./tp4check.sh check-27b-final ./trashcan check; ./tp4check.sh checkmtp-27b-final ./trashcan checkmtp'" 2>&1 | grep -v '^\[clocks\]'
