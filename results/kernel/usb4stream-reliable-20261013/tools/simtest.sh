#!/bin/bash
# Run a window script locally with every remote command simulated (sim/ssh, gpuq, rsync, runners), keeping its real
# heartbeat loop, background jobs, waits and traps: it must reach the end and exit 0 with no process left behind; with
# TERM_AFTER=S, a TERM after S seconds must end it (exit 130) with its cleanup run and no process left behind.
# usage: simtest.sh SCRIPT [ENV=VALUE...]      e.g. simtest.sh window2.sh DEPLOYED=1
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
script=$1
shift
d=$(mktemp -d "$HOME/dev/claude-work/flashnext-20261012/stream/sim.XXXXXX")
mkdir -p "$d/bin"
cp "$here/sim/ssh" "$d/bin/ssh"
printf '#!/bin/sh\necho "$(date +%%s) gpuq $*" >> "$SIMLOG"\n' > "$d/bin/gpuq"
printf '#!/bin/sh\necho "$(date +%%s) rsync $*" >> "$SIMLOG"\n' > "$d/bin/rsync"
printf '#!/bin/sh\necho "cool after 0 s"\n' > "$d/bin/cool.sh"
chmod +x "$d/bin/"*
export SIMLOG=$d/remote.log PATH=$d/bin:$PATH Q4RUN=$here/sim/runner.py TPRUN=$here/sim/runner.py COOL=$d/bin/cool.sh
export TICKETS=incite=1,indite=2,inflight=3,alight=4,invite=5,enlight=6 OUT=$d/out
for kv in "$@"; do export "${kv?}"; done
start=$SECONDS
bash "$here/$script" > "$d/stdout.txt" 2>&1 &
pid=$!
if [ -n "${TERM_AFTER:-}" ]; then
    sleep "$TERM_AFTER"
    kill -TERM $pid
fi
wait $pid
rc=$?
sleep 1
left=$(pgrep -f "$d/bin|$here/sim/" | grep -v "^$$\$" | tr '\n' ' ')
leftsh=$(pgrep -f "bash $here/$script" | tr '\n' ' ')
echo "simtest $script: rc $rc after $((SECONDS - start)) s; processes left: [${left}${leftsh}]"
echo "  last steps: $(grep -E '^\[' "$d/out/steps.log" 2>/dev/null | tail -2 | tr '\n' '|')"
echo "  cleanup seen: $(grep -c 'systemctl start tblink' "$SIMLOG" 2>/dev/null) tblink starts, $(grep -c 'pkill' "$SIMLOG" 2>/dev/null) pkills"
echo "  dir: $d"
