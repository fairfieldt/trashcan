#!/bin/bash
# Thunderbolt and netconsole checks on one Mac Pro (no GPU use) for the 7.2.9-p2p -> 7.3 A/B: tblink's links and
# streams, the stream configfs attributes (7.3 adds busy_poll), the /dev/tbstream* permissions from the udev rule,
# thunderbolt_net's throttling parameter and ping over each link, and a netconsole token.
# usage: ssh MAC 'cd ~/trashcan-exp/kernel73 && ./tb.sh [TAG]'    -> ./HOST-$(uname -r)[-TAG]/tb.txt
# then on the workstation: grep "TOKEN" ~/dev/claude-work/trashcan-kbuild/netconsole-incite.log
set -uo pipefail
cd "$(dirname "$0")"
R=$PWD/$(hostname)-$(uname -r)${1:+-$1}
cfg=/sys/kernel/config/thunderbolt/stream
mkdir -p "$R"
{
    echo "== tblink status"; sudo -n /usr/local/sbin/tblink status
    echo "== tblink links"; sudo -n /usr/local/sbin/tblink links
    echo "== thunderbolt_net throttling=$(cat /sys/module/thunderbolt_net/parameters/throttling)"
    echo "== stream groups"
    for g in "$cfg"/*/*/; do
        echo "$g: $(ls "$g" | xargs)"
        for a in busy_poll index in_hopid out_hopid ring_size throttling; do
            [ -e "$g/$a" ] && echo "  $a=$(sudo -n cat "$g/$a")"
        done
    done
    ls -la /dev/tbstream*
    echo "== ping over each link (100 x 10 ms)"
    sudo -n /usr/local/sbin/tblink links | awk 'NR > 1 && $6 == "up" {print $9, $7}' | while read -r peer iface; do
        echo "$iface -> $peer: $(ping -c 100 -i 0.01 -q -I "$iface" "$peer" | tail -1)"
    done
    tok="kernel73 netconsole check $(hostname) $(uname -r) $(date +%s)"
    echo "<6>$tok" | sudo -n tee /dev/kmsg > /dev/null
    echo "netconsole token: $tok"
} > "$R/tb.txt" 2>&1
cat "$R/tb.txt"
