#!/bin/bash
# Orchestrate one 27B TP4 bench from the Mac: restart the node on invite with ENV, then run the driver on incite.
# usage: tp4run.sh NAME INCITE_LEASE INVITE_LEASE "ENV ASSIGNMENTS" [mtp]
name=$1; li=$2; lv=$3; envs=$4; mtp=$5
M='~/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf'
MTP='~/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf'
S=/dev/tbstream0,/dev/tbstream1,/dev/tbstream2
nodemtp=""; drvmtp=""
if [ "$mtp" = mtp ]; then nodemtp="--mtp $MTP"; drvmtp="--mtp $MTP"; fi
ssh invite "pkill -f '^[.]/trashcan(-base)? qwen-tp-node'; sleep 2; cd ~/trashcan-exp/decode && env $envs gpuq run $lv -- ./node.sh node-$name ./trashcan $M $nodemtp --stream $S" > /dev/null
sleep 2
ssh incite "cd ~/trashcan-exp/decode && env $envs gpuq run $li -- ./tp4.sh $name ./trashcan $drvmtp" 2>&1 | grep -v '^\[clocks\]'
