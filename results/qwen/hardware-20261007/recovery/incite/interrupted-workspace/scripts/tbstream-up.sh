#!/bin/bash
# Create one USB4STREAM named NAME on every Thunderbolt link between nodes A and B (run from the Mac) and print,
# per link in A's bus order, A's and B's /dev/tbstreamX. A allocates the HopIDs; B gets the complements.
# Loads thunderbolt_stream on both. Remove with tbstream-down.sh.
# usage: tbstream-up.sh A B [NAME]
set -euo pipefail
a=$1 b=$2 name=${3:-xmac}
[[ $name =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'invalid stream name' >&2; exit 1; }
cfg=/sys/kernel/config/thunderbolt/stream
for h in "$a" "$b"; do ssh "$h" 'sudo modprobe thunderbolt_stream'; done
# tblink links: xdomain peer local_bus peer_bus lanes state iface address peer_address
links_a=$(ssh "$a" /usr/local/sbin/tblink links | awk -F'\t' -v p="$b" 'NR>1 && $2==p && $6=="up" {print $3, $4, $1}' | sort -n)
links_b=$(ssh "$b" /usr/local/sbin/tblink links | awk -F'\t' -v p="$a" 'NR>1 && $2==p && $6=="up" {print $3, $4, $1}')
[ -n "$links_a" ] || { echo "no working links between $a and $b" >&2; exit 1; }
stream_service() {
  ssh -n "$1" "for s in /sys/bus/thunderbolt/devices/$2.*; do
    [ \"\$(basename \"\$(readlink \"\$s/driver\")\")\" = thunderbolt_stream ] && basename \"\$s\"
  done; true"
}
while read -r bus peer_bus xa; do
  xb=$(awk -v l="$peer_bus" -v p="$bus" '$1==l && $2==p {print $3}' <<<"$links_b")
  [ -n "$xb" ] || { echo "no matching link on $b for $a bus $bus" >&2; exit 1; }
  # Discover by bound driver: net and stream indices can swap after reconnect.
  sa=$(stream_service "$a" "$xa") sb=$(stream_service "$b" "$xb")
  [ -n "$sa" ] && [ -n "$sb" ] || { echo "missing stream service on bus $bus" >&2; exit 1; }
  read -r in_a out_a dev_a < <(ssh -n "$a" "d=$cfg/$sa/$name; sudo mkdir -p \$d
    [ \"\$(sudo cat \$d/in_hopid)\" -gt 0 ] || echo -1 | sudo tee \$d/in_hopid \$d/out_hopid >/dev/null
    echo \$(sudo cat \$d/in_hopid) \$(sudo cat \$d/out_hopid) \$(sudo cat \$d/index)")
  dev_b=$(ssh -n "$b" "d=$cfg/$sb/$name; sudo mkdir -p \$d
    echo $out_a | sudo tee \$d/in_hopid >/dev/null; echo $in_a | sudo tee \$d/out_hopid >/dev/null; sudo cat \$d/index")
  echo "bus $bus: $a $sa/$name /dev/tbstream$dev_a (in $in_a out $out_a) <-> $b $sb/$name /dev/tbstream$dev_b"
done <<<"$links_a"
