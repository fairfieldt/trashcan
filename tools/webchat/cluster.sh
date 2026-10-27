#!/bin/bash
# Start or switch the model the demo serves, from the cluster's first Mac (incite), under gpuq leases. Uses the
# launcher and inventory in ~/trashcan-exp/cluster (deploy/trashcan-cluster, cluster.toml); server.py's POST /api/switch runs it.
#
#   cluster.sh up RECIPE    stop what runs, lease RECIPE's Macs, start it: 6-flashnext (six Macs) or 4-qwen27b (four,
#                           with MTP)
#   cluster.sh down         stop it and release the leases
#   cluster.sh status       the launcher's status; cluster.sh current prints the running recipe
set -uo pipefail
C=$HOME/trashcan-exp/cluster
LAUNCH=(python3 -u "$C/deploy/trashcan-cluster" -c "$C/cluster.toml")
S=$C/state
mkdir -p "$S"
exec 9> "$S/lock"
flock -w 600 9 || { echo "another switch is still running" >&2; exit 1; }

# In role order along the cables (cluster.toml): the six-Mac chain, and the four-Mac loop as machines 0-1, 0-2, 1-3, 2-3.
nodes_of() {
    case $1 in
    6-flashnext) echo "incite inflight alight indite invite enlight" ;;
    4-qwen27b) echo "incite inflight indite alight" ;;
    *) return 1 ;;
    esac
}

down() {
    "${LAUNCH[@]}" down 2>&1 | tail -1
    if [ -f "$S/tickets" ]; then
        for t in $(cat "$S/tickets"); do gpuq --on "${t%%=*}" release "${t#*=}" -r "demo: stopped" > /dev/null 2>&1; done
    fi
    rm -f "$S/tickets" "$S/current"
}

case ${1:-} in
up)
    recipe=${2:?usage: $0 up RECIPE}
    nodes=$(nodes_of "$recipe") || { echo "unknown recipe $recipe (6-flashnext or 4-qwen27b)" >&2; exit 2; }
    down
    : > "$S/tickets"
    for m in $nodes; do
        id=$(gpuq --on "$m" request "demo $recipe" --est 120 --owner "trashcan demo" | sed -n 's/^Ticket #\([0-9]*\) created.*/\1/p')
        [ -n "$id" ] || { echo "lease request on $m failed" >&2; down; exit 1; }
        echo "$m=$id" >> "$S/tickets"
    done
    echo "waiting for the leases"
    # All at once, every 10 s: a status call counts as a heartbeat, so a lease granted early is not reclaimed as idle
    # while another Mac is still busy. Gives up after 10 min, saying what blocks.
    python3 - "$S/tickets" <<'PY' || { down; exit 1; }
import json, subprocess, sys, time
tickets = [l.strip().split('=') for l in open(sys.argv[1]) if l.strip()]
start = time.time()
while True:
    waiting = []
    for m, i in tickets:
        out = subprocess.run(['gpuq', '--on', m, 'status', i, '--json'], capture_output=True, text=True).stdout
        d = json.loads(out or '{}')
        you = d.get('you') or {}
        if you.get('end_reason') or you.get('ended_at'):
            sys.exit(f"lease on {m} ended: {you.get('end_reason')}")
        if you.get('state') != 'running':
            waiting.append(f"{m} ({you.get('state')}{': ' + str(d['blocked']) if d.get('blocked') else ''})")
    if not waiting:
        sys.exit(0)
    if time.time() - start > 600:
        sys.exit('leases not granted after 10 min: ' + ', '.join(waiting))
    if int(time.time() - start) % 30 < 10:
        print('waiting for ' + ', '.join(waiting), flush=True)
    time.sleep(10)
PY
    W=(); for t in $(cat "$S/tickets"); do W+=(--wrap "${t%%=*}=gpuq run ${t#*=} --"); done
    extra=(); [ "$recipe" = 4-qwen27b ] && extra=(--mtp)
    # Demo only, not the published recipe: the 27B with the most context the D700s hold in VRAM (204,800 tokens at
    # 17 KB per token per GPU: 5,669 MiB each, decode as fast as at 8K; 262,144 spilled past VRAM).
    [ "$recipe" = 4-qwen27b ] && python3 - "$C/deploy/recipes.toml" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
i = s.index('[recipe.4-qwen27b]'); j = s.index('driver =', i); k = s.index('\n', j)
s = s[:j] + 'driver = ["serve", "{qwen38-27b}", "--port", "{serve_port}", "--context", "204800"]' + s[k:]
open(p, 'w').write(s)
PY
    if ! "${LAUNCH[@]}" up "$recipe" --nodes "$(echo $nodes | tr ' ' ,)" "${extra[@]}" "${W[@]}"; then
        echo "starting $recipe failed" >&2; down; exit 1
    fi
    echo "$recipe" > "$S/current"
    echo "ready: $recipe"
    ;;
down) down; echo "stopped" ;;
status) "${LAUNCH[@]}" status ;;
current) cat "$S/current" 2>/dev/null ;;
*) echo "usage: $0 up RECIPE | down | status | current" >&2; exit 2 ;;
esac
