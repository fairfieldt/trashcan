# usage: tp8.sh NAME BINARY 'DRIVER COMMAND ARGS' [ENV]
T=${TICKETS:?}
python3 ~/dev/claude-work/trashcan-20261007/tprun.py --name "$1" --binary "$2" --tickets $T --env "${4:-}" \
  --node 'indite:--machine 3 --peer 1=/dev/tbstream0 --peer 2=/dev/tbstream1' \
  --node 'enlight:--machine 2 --peer 0=/dev/tbstream0 --peer 3=/dev/tbstream2' \
  --node 'invite:--machine 1 --peer 0=/dev/tbstream0 --peer 3=/dev/tbstream2' \
  --driver "$3 --tp-peer 1=/dev/tbstream0 --tp-peer 2=/dev/tbstream1"
