#!/bin/bash
# From this Mac: install the repo's fancurve.sh, fancurve.service and system-crash-capture.py on each HOST (only these,
# as docs/hardware.md says; the full `provision.sh HOST system` restarts unrelated services) and restart the two
# services. Refuses a host whose installed copies differ from the previous commit's (a local change we'd overwrite),
# unless FORCE=1. Prints the fan state afterwards.
# usage: results/kernel/linux-7.3-20261014/install-fan.sh HOST...
set -euo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd)
base=${BASE:-7de407f1~1}    # the commit before the hwmon-layout fix
want_old=$(cd "$repo" && for f in scripts/fancurve.sh scripts/fancurve.service scripts/system-crash-capture.py; do
    git show "$base:$f" | shasum -a 256 | cut -d' ' -f1; done | xargs)
for h in "$@"; do
    have=$(ssh "$h" 'sha256sum /usr/local/sbin/fancurve.sh /etc/systemd/system/fancurve.service /usr/local/sbin/system-crash-capture.py' |
        cut -d' ' -f1 | xargs)
    new=$(cd "$repo" && shasum -a 256 scripts/fancurve.sh scripts/fancurve.service scripts/system-crash-capture.py | cut -d' ' -f1 | xargs)
    if [ "$have" != "$want_old" ] && [ "$have" != "$new" ] && [ "${FORCE:-0}" != 1 ]; then
        echo "$h: installed files differ from $base's and from the new ones; not touching it (FORCE=1 to override)" >&2
        continue
    fi
    ssh "$h" 'rm -rf ~/.cache/kernel73-fan && mkdir -p ~/.cache/kernel73-fan'
    scp -q "$repo/scripts/fancurve.sh" "$repo/scripts/fancurve.service" "$repo/scripts/system-crash-capture.py" "$h":.cache/kernel73-fan/
    ssh "$h" 'set -e; cd ~/.cache/kernel73-fan
        sudo -n install -m 0755 fancurve.sh /usr/local/sbin/fancurve.sh
        sudo -n install -m 0644 fancurve.service /etc/systemd/system/fancurve.service
        sudo -n install -m 0755 system-crash-capture.py /usr/local/sbin/system-crash-capture.py
        sudo -n systemctl daemon-reload
        sudo -n systemctl restart fancurve.service system-crash-capture.service
        cd && rm -rf ~/.cache/kernel73-fan
        sleep 4
        smc=/sys/devices/platform/applesmc.768; fan=$smc; [ -e $smc/fan1_manual ] || fan=$(echo $smc/hwmon/hwmon*)
        echo "$(hostname) $(uname -r): fancurve $(systemctl is-active fancurve), system-crash-capture $(systemctl is-active system-crash-capture), fan $(cat $fan/fan1_input) rpm; $(sudo -n journalctl -u fancurve -n 1 -q -o cat)"'
done
