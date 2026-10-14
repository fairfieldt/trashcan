#!/bin/bash
# Install the automatic stream setup on Mac Pros, as nodes/macpro/provision.sh's `system` step installs root/ (staged
# under ~/.cache/trashcan-node, installed as root): /usr/local/sbin/tblink and /etc/udev/rules.d/70-tbstream.rules;
# reload udev and restart tblink.service, then list each Mac's links with their streams (tblink creates them within
# its 5 s loop). Touches Thunderbolt state: run inside a six-Mac lease.
# usage (from the repo): results/kernel/usb4stream-reliable-20261013/tools/deploy-setup.sh HOST...
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
root=$repo/nodes/macpro/root
[ "$#" -ge 1 ] || { echo "usage: deploy-setup.sh HOST..." >&2; exit 2; }
rc=0
for h in "$@"; do
    ssh "$h" 'rm -rf ~/.cache/trashcan-node && mkdir -p ~/.cache/trashcan-node'
    rsync -a "$root/usr/local/sbin/tblink" "$root/etc/udev/rules.d/70-tbstream.rules" "$h:.cache/trashcan-node/"
    ssh "$h" 'set -e
        s=~/.cache/trashcan-node
        sudo -n install -m 0755 -o root -g root $s/tblink /usr/local/sbin/tblink
        sudo -n install -D -m 0644 -o root -g root $s/70-tbstream.rules /etc/udev/rules.d/70-tbstream.rules
        rm -rf $s
        sudo -n udevadm control --reload
        sudo -n udevadm trigger --action=change --sysname-match="tbstream*"
        sudo -n systemctl restart tblink.service
        echo "$(hostname): tblink $(md5sum /usr/local/sbin/tblink | cut -c1-12), rule $(md5sum /etc/udev/rules.d/70-tbstream.rules | cut -c1-12), $(systemctl is-active tblink)"' || rc=1
done
sleep 8
for h in "$@"; do
    echo "== $h"
    ssh "$h" '/usr/local/sbin/tblink links | tail -n +2 | cut -f2,5,6,10; ls -l /dev/tbstream* 2>/dev/null | awk "{print \$1, \$4, \$NF}"'
done
exit $rc
