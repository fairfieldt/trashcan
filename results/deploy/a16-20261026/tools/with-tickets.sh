#!/bin/bash
# Lab only: run CMD with --wrap NODE='gpuq run ID --' for every NODE=ID in $TICKETS (chain6.sh exports it).
w=(); IFS=, read -ra t <<< "$TICKETS"; for x in "${t[@]}"; do w+=(--wrap "${x%%=*}=gpuq run ${x#*=} --"); done
"$@" "${w[@]}"
