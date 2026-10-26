#!/bin/bash
# Lab only: hold gpuq leases on some Macs for one command, like chain6.sh but for a subset. Requests go in chain6's
# fixed machine order under chain6's lock (no interleaving with six-Mac sessions), then chain6-wait.py waits and
# heartbeats. CMD gets WRAPS="--wrap incite=gpuq run ID -- ..." for deploy/trashcan-cluster.
# usage: lease.sh LABEL EST_MIN MACHINE,... -- CMD...
set -uo pipefail
label=$1 est=$2 want=,$3,
shift 3; [ "${1:-}" = -- ] && shift
fn=$HOME/dev/claude-work/flashnext-20261012
tickets=()
release() { for t in "${tickets[@]}"; do gpuq --on "${t%%=*}" release "${t#*=}" -r "$label: $1" > /dev/null 2>&1; done; }
trap 'release "stopped"; exit 130' INT TERM
exec 9> "$fn/chain6.lock"
flock 9
for m in incite indite inflight alight invite enlight; do
    [[ $want == *,$m,* ]] || continue
    id=$(gpuq --on $m request "$label" --est "$est" --owner "A16 deployment" | sed -n 's/^Ticket #\([0-9]*\) created.*/\1/p')
    [ -n "$id" ] || { echo "lease: request on $m failed" >&2; release "request failed"; exit 1; }
    tickets+=("$m=$id")
done
exec 9>&-
echo "lease: queued ${tickets[*]}" >&2
python3 "$fn/chain6-wait.py" "${tickets[@]}" || { release "a ticket ended while waiting"; exit 1; }
WRAPS=()
for t in "${tickets[@]}"; do WRAPS+=(--wrap "${t%%=*}=gpuq run ${t#*=} --"); done
export TICKETS=$(IFS=,; echo "${tickets[*]}")
echo "lease: holding $TICKETS" >&2
"$@" "${WRAPS[@]}"
rc=$?
trap - INT TERM
release "(rc $rc) A16 deployment acceptance step"
exit $rc
