#!/bin/bash
# llama.cpp b8e9ce8 on the M3 Ultra's CPU: G-iq4 trunk with the MTP GGUF as draft-mtp.
# usage: llama_server.sh MTP_GGUF PORT LOG
exec /Users/fairfieldt/dev/trashcan/llama.cpp/build-mac/bin/llama-server \
    -m /Volumes/Green/Qwen/flash-next/cand/G-iq4/G-iq4-00001-of-00006.gguf \
    -md "$1" --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 \
    -dev none -devd none -t 24 -c 8192 -np 1 --port "$2" --no-webui > "$3" 2>&1
