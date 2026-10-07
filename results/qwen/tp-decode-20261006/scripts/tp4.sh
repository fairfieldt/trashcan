#!/bin/bash
# Driver side of a 27B TP4 run over USB4STREAM. usage: tp4.sh NAME BINARY [extra qwen-bench args...]
name=$1; bin=$2; shift 2
M=~/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf
S=/dev/tbstream0,/dev/tbstream1,/dev/tbstream2
cd ~/trashcan-exp/decode && ./bench.sh $name $bin qwen-bench $M --tp-stream $S --pp 512 --tg 128 "$@"
