#!/bin/bash
# Run a window script locally with every remote command simulated, keeping its real heartbeat loop, background jobs,
# waits and traps. sim/ssh runs each remote snippet with bash in a simulated Mac (its home in $d/home/HOST, made from
# sim/home; the Mac's own programs stubbed in sim/stubs) and, like ssh, returns only once nothing holds the snippet's
# output, so a snippet that leaves a background job holding it hangs here as on the Macs. gpuq and rsync are logged
# (`gpuq run ID -- CMD` runs CMD).
# The script must reach the end and exit 0 with no process left behind, within SIM_TIMEOUT s (default 180; past it,
# the run is reported as hung and stopped); with TERM_AFTER=S, a TERM after S seconds must end it (exit 130) with its
# cleanup run and nothing left behind.
# usage: simtest.sh SCRIPT [ENV=VALUE...]      e.g. simtest.sh churn-session.sh ARGS="invite enlight 20"
#   (ARGS: the script's arguments)
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
script=$1
shift
d=$(mktemp -d "$HOME/dev/claude-work/flashnext-20261012/stream/sim.XXXXXX")
mkdir -p "$d/bin" "$d/home"
for h in incite indite inflight alight invite enlight; do cp -R "$here/sim/home" "$d/home/$h"; done
cp "$here/sim/ssh" "$d/bin/ssh"
# gpuq: logged; `gpuq run ID -- CMD` (on a simulated Mac) runs CMD.
printf '#!/bin/sh\necho "$(date +%%s) gpuq $*" >> "$SIMLOG"\nif [ "$1" = run ]; then shift 2; [ "$1" = -- ] && shift; exec "$@"; fi\n' > "$d/bin/gpuq"
printf '#!/bin/sh\necho "$(date +%%s) rsync $*" >> "$SIMLOG"\n' > "$d/bin/rsync"
printf '#!/bin/sh\necho "cool after 0 s"\n' > "$d/bin/cool.sh"
chmod +x "$d/bin/"*
export SIMLOG=$d/remote.log SIMHOME=$d/home SIMSTUBS=$here/sim/stubs PATH=$d/bin:$PATH
export Q4RUN=$here/sim/runner.py TPRUN=$here/sim/runner.py COOL=$d/bin/cool.sh
export TICKETS=incite=1,indite=2,inflight=3,alight=4,invite=5,enlight=6 OUT=$d/out
for kv in "$@"; do export "${kv?}"; done
start=$SECONDS
bash "$here/$script" ${ARGS:-} > "$d/stdout.txt" 2>&1 &
pid=$!
hung=""
if [ -n "${TERM_AFTER:-}" ]; then
    sleep "$TERM_AFTER"
    kill -TERM $pid
fi
while kill -0 $pid 2> /dev/null; do
    if [ $((SECONDS - start)) -ge "${SIM_TIMEOUT:-180}" ]; then
        hung=" HUNG after ${SIM_TIMEOUT:-180} s (last remote command: $(tail -1 "$SIMLOG" | cut -c1-200))"
        # The script runs its TERM trap only once its stuck command returns: end the stuck ssh stand-ins too.
        kill -TERM $pid
        pkill -f "$d/bin/ssh"
        for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 $pid 2> /dev/null || break; sleep 1; pkill -f "$d/bin/ssh"; done
        kill -9 $pid 2> /dev/null
        break
    fi
    sleep 1
done
wait $pid
rc=$?
sleep 1
left=$(pgrep -f "$d/bin|$d/home|$here/sim/" | grep -v "^$$\$" | tr '\n' ' ')
leftsh=$(pgrep -f "bash $here/$script" | tr '\n' ' ')
echo "simtest $script: rc $rc after $((SECONDS - start)) s;$hung processes left: [${left}${leftsh}]"
echo "  last steps: $(grep -E '^\[' "$d/out/steps.log" 2>/dev/null | tail -2 | tr '\n' '|')"
echo "  dir: $d"
[ -z "$left$leftsh" ] || pkill -f "$d/home" 2> /dev/null
