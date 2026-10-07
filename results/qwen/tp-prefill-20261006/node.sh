#!/bin/bash
# The other machine's ranks over the three xmac streams, in the foreground, with a clock log.
# usage: [ENV="VAR=1 ..."] node.sh NAME MODEL [ARGS...]   (ARGS e.g. --mtp FILE)
set -uo pipefail
cd ~/trashcan-exp/prefill
name=$1; shift
out=results/tp-prefill
mkdir -p $out/clocks
echo "$(date -Is) node $name: ${ENV:-} $*" >> $out/commands.log
exec scripts/with-clocks.sh $out/clocks/$name.txt env ${ENV:-} ./trashcan qwen-tp-node "$@" --stream /dev/tbstream0,/dev/tbstream1,/dev/tbstream2 > $out/$name.log 2>&1
