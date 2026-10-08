#!/bin/bash
# One TP8 measurement over the four Macs (cabling of results/qwen/tp8-20261007): nodes first, then the driver on
# incite, every process under its gpuq lease, scripts/with-clocks.sh and the 0 us latency request. ENV goes to every
# process, nodes included (the kernel switches are per process; TRASHCAN_TP_WIRE is the driver's and reaches the nodes
# at setup). Extra tprun.py options (e.g. --profile) follow ENV.
# usage: TICKETS=incite=ID,invite=ID,enlight=ID,indite=ID tp8.sh NAME BINARY 'DRIVER COMMAND ARGS' ['ENV'] [OPTIONS]
T=${TICKETS:?set TICKETS=incite=ID,invite=ID,enlight=ID,indite=ID}
here=$(dirname "$0")
name=$1 binary=$2 driver=$3 env=${4:-}
shift 4 2>/dev/null || shift $#
python3 "$here/tprun.py" --name "$name" --binary "$binary" --tickets "$T" --env "$env" "$@" \
  --node 'indite:--machine 3 --peer 1=/dev/tbstream0 --peer 2=/dev/tbstream1' \
  --node 'enlight:--machine 2 --peer 0=/dev/tbstream0 --peer 3=/dev/tbstream2' \
  --node 'invite:--machine 1 --peer 0=/dev/tbstream0 --peer 3=/dev/tbstream2' \
  --driver "$driver --tp-peer 1=/dev/tbstream0 --tp-peer 2=/dev/tbstream1"
