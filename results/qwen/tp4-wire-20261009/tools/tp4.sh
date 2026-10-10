#!/bin/bash
# One TP4 measurement over incite (driver, machine 0) and invite (machine 1) on the one incite-invite cable: the tp8
# stream of cable 1 (results/qwen/tp8-mesh-20261009), /dev/tbstream0 on both Macs. The node starts first, then the
# driver, each under its gpuq lease, scripts/with-clocks.sh and the 0 us latency request (tprun.py of
# results/qwen/tp8-opt-20261008). ENV goes to both processes; TRASHCAN_TP_WIRE is the driver's and reaches the node
# at setup.
# usage: TICKETS=incite=ID,invite=ID tp4.sh NAME 'DRIVER COMMAND ARGS' ['ENV']
T=${TICKETS:?set TICKETS=incite=ID,invite=ID}
here=$(dirname "$0")
name=$1 driver=$2 env=${3:-}
TP8_ROOT=/home/fairfieldt/trashcan-exp/tp4wire python3 "$here/../../tp8-opt-20261008/tools/tprun.py" \
  --name "$name" --binary trashcan --tickets "$T" --env "$env" \
  --node 'invite:--stream /dev/tbstream0' \
  --driver "$driver --tp-stream /dev/tbstream0"
