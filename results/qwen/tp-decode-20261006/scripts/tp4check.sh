#!/bin/bash
# Driver side of the 27B TP4 correctness runs. usage: tp4check.sh NAME BINARY validate|check|checkmtp
name=$1; bin=$2; what=$3
M=~/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf
MTP=~/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf
S=/dev/tbstream0,/dev/tbstream1,/dev/tbstream2
cd ~/trashcan-exp/decode
case $what in
validate) scripts/with-clocks.sh results/clocks/$name.txt $bin qwen-validate $M results/$name --tp-stream $S > results/$name.out 2>&1 ;;
check) scripts/with-clocks.sh results/clocks/$name.txt $bin qwen-check $M --tp-stream $S --verify 3 --tokens 600 --steps 128 > results/$name.json 2> results/$name.err ;;
checkmtp) scripts/with-clocks.sh results/clocks/$name.txt $bin qwen-check $M --tp-stream $S --mtp $MTP --tokens 600 --steps 256 > results/$name.json 2> results/$name.err ;;
esac
echo "$name rc=$?"
