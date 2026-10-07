#!/bin/bash
# Create one USB4STREAM named NAME on every working link between A and B (run from the Mac).
# Print endpoints in A's bus order. A allocates the HopIDs; B gets the complements.
# Existing streams must already agree; their HopIDs are never rewritten.
# usage: tbstream-up.sh A B [NAME]
set -euo pipefail
[ "$#" -ge 2 ] && [ "$#" -le 3 ] || { echo 'usage: tbstream-up.sh A B [NAME]' >&2; exit 1; }
a=$1 b=$2 name=${3:-xmac}
[[ $name =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'invalid stream name' >&2; exit 1; }
# Quote positional arguments for the remote shell; the script itself travels on stdin.
remote_script() {
  local host=$1 command
  shift
  printf -v command '%q ' bash -s -- "$@"
  ssh "$host" "$command"
}
for h in "$a" "$b"; do ssh "$h" 'sudo -n modprobe thunderbolt_stream'; done
# tblink links: xdomain peer local_bus peer_bus lanes state iface address peer_address
links_a=$(ssh "$a" /usr/local/sbin/tblink links | awk -F'\t' -v p="$b" 'NR>1 && $2==p && $6=="up" {print $3, $4, $1}' | sort -n)
links_b=$(ssh "$b" /usr/local/sbin/tblink links | awk -F'\t' -v p="$a" 'NR>1 && $2==p && $6=="up" {print $3, $4, $1}')
[ -n "$links_a" ] || { echo "no working links between $a and $b" >&2; exit 1; }
stream_service() {
  remote_script "$1" "$2" <<'REMOTE'
set -euo pipefail
xdomain=$1
[[ $xdomain =~ ^[0-9]+-[0-9a-fA-F]+$ ]] || { echo "invalid XDomain: $xdomain" >&2; exit 1; }
# Stream service advertisement is asynchronous after module loading or reconnect.
for ((attempt=0; attempt<=15; attempt++)); do
  matches=()
  for service in /sys/bus/thunderbolt/devices/"$xdomain".*; do
    driver=$(readlink "$service/driver" 2>/dev/null || true)
    [ "${driver##*/}" != thunderbolt_stream ] || matches+=("${service##*/}")
  done
  if [ "${#matches[@]}" -eq 1 ]; then printf '%s\n' "${matches[0]}"; exit 0; fi
  if [ "${#matches[@]}" -gt 1 ]; then echo "multiple stream services for $xdomain" >&2; exit 1; fi
  [ "$attempt" -eq 15 ] || sleep 1
done
echo "missing stream service for $xdomain after 15 seconds" >&2
exit 1
REMOTE
}
configure_stream() {
  remote_script "$1" "$2" "$name" "${3:-allocate}" "${4:-}" <<'REMOTE'
set -euo pipefail
service=$1 name=$2 want_in=$3 want_out=$4
[[ $service =~ ^[0-9]+-[0-9a-fA-F]+\.[0-9]+$ ]] || { echo "invalid stream service: $service" >&2; exit 1; }
d=/sys/kernel/config/thunderbolt/stream/$service/$name
existing=0
if sudo -n test -d "$d"; then
  existing=1
else
  sudo -n mkdir -p "${d%/*}"
  sudo -n mkdir "$d"
fi
in=$(sudo -n cat "$d/in_hopid")
out=$(sudo -n cat "$d/out_hopid")
[[ $in =~ ^-?[0-9]+$ && $out =~ ^-?[0-9]+$ ]] || { echo "invalid HopIDs in $d" >&2; exit 1; }
if [ "$want_in" = allocate ]; then
  if [ "$existing" -eq 1 ] && { [ "$in" -le 0 ] || [ "$out" -le 0 ]; }; then
    echo "existing stream $d has incomplete HopIDs ($in, $out); left unchanged" >&2
    exit 1
  fi
  [ "$in" -gt 0 ] || printf '%s\n' -1 | sudo -n tee "$d/in_hopid" >/dev/null
  [ "$out" -gt 0 ] || printf '%s\n' -1 | sudo -n tee "$d/out_hopid" >/dev/null
else
  [[ $want_in =~ ^[0-9]+$ && $want_out =~ ^[0-9]+$ ]] && [ "$want_in" -gt 0 ] && [ "$want_out" -gt 0 ] || { echo 'invalid requested HopIDs' >&2; exit 1; }
  if [ "$existing" -eq 1 ] && { [ "$in" != "$want_in" ] || [ "$out" != "$want_out" ]; }; then
    echo "existing stream $d has HopIDs ($in, $out), expected ($want_in, $want_out); left unchanged" >&2
    exit 1
  fi
  [ "$in" = "$want_in" ] || printf '%s\n' "$want_in" | sudo -n tee "$d/in_hopid" >/dev/null
  [ "$out" = "$want_out" ] || printf '%s\n' "$want_out" | sudo -n tee "$d/out_hopid" >/dev/null
fi
in=$(sudo -n cat "$d/in_hopid")
out=$(sudo -n cat "$d/out_hopid")
index=$(sudo -n cat "$d/index")
[[ $in =~ ^[0-9]+$ && $out =~ ^[0-9]+$ && $index =~ ^[0-9]+$ ]] && [ "$in" -gt 0 ] && [ "$out" -gt 0 ] || { echo "stream $d did not allocate valid HopIDs and device index" >&2; exit 1; }
if [ "$want_in" != allocate ] && { [ "$in" != "$want_in" ] || [ "$out" != "$want_out" ]; }; then
  echo "stream $d did not retain requested HopIDs" >&2
  exit 1
fi
device=/dev/tbstream$index
for ((attempt=0; attempt<=15; attempt++)); do
  [ ! -c "$device" ] || break
  [ "$attempt" -eq 15 ] || sleep 1
done
[ -c "$device" ] || { echo "missing character device $device after 15 seconds" >&2; exit 1; }
# Scope access to this exact endpoint and the SSH user's primary group. Leave
# permissions alone when the user already has access (including existing ACLs).
if [ ! -r "$device" ] || [ ! -w "$device" ]; then
  sudo -n chgrp -- "$(id -gn)" "$device"
  sudo -n chmod 660 -- "$device"
fi
[ -r "$device" ] && [ -w "$device" ] || { echo "cannot access $device as $(id -un)" >&2; exit 1; }
printf '%s %s %s\n' "$in" "$out" "$index"
REMOTE
}
while read -r bus peer_bus xa; do
  xb=$(awk -v l="$peer_bus" -v p="$bus" '$1==l && $2==p {print $3}' <<<"$links_b")
  [ -n "$xb" ] || { echo "no matching link on $b for $a bus $bus" >&2; exit 1; }
  # Discover by bound driver: net and stream indices can swap after reconnect.
  sa=$(stream_service "$a" "$xa")
  sb=$(stream_service "$b" "$xb")
  # Keep command substitutions separate from read so SSH failures propagate.
  result_a=$(configure_stream "$a" "$sa")
  read -r in_a out_a dev_a <<<"$result_a"
  result_b=$(configure_stream "$b" "$sb" "$out_a" "$in_a")
  read -r in_b out_b dev_b <<<"$result_b"
  echo "bus $bus: $a $sa/$name /dev/tbstream$dev_a (in $in_a out $out_a) <-> $b $sb/$name /dev/tbstream$dev_b"
done <<<"$links_a"
