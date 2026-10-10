#!/bin/bash
# tp8.sh with all six cables: every machine names its three peers, so the same sessions can run the ring's all-reduces
# (default) or the full mesh's (TRASHCAN_TP_MESH=full in ENV). Streams: tbstreamN numbered as in streams-6.txt.
# usage: TICKETS=incite=ID,invite=ID,enlight=ID,indite=ID tp8full.sh NAME BINARY 'DRIVER COMMAND ARGS' ['ENV'] [OPTIONS]
T=${TICKETS:?set TICKETS=incite=ID,invite=ID,enlight=ID,indite=ID}
here=$(dirname "$0")
name=$1 binary=$2 driver=$3 env=${4:-}
shift 4 2>/dev/null || shift $#
python3 "$here/../../tp8-opt-20261008/tools/tprun.py" --name "$name" --binary "$binary" --tickets "$T" --env "$env" "$@" \
  --node 'indite:--machine 3 --peer 1=/dev/tbstream0 --peer 2=/dev/tbstream1 --peer 0=/dev/tbstream2' \
  --node 'enlight:--machine 2 --peer 0=/dev/tbstream0 --peer 1=/dev/tbstream1 --peer 3=/dev/tbstream2' \
  --node 'invite:--machine 1 --peer 0=/dev/tbstream0 --peer 2=/dev/tbstream1 --peer 3=/dev/tbstream2' \
  --driver "$driver --tp-peer 1=/dev/tbstream0 --tp-peer 2=/dev/tbstream1 --tp-peer 3=/dev/tbstream2"
