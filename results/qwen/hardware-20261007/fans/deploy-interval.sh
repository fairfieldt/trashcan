#!/bin/bash
# Deploy only the validated interval guard after hardware measurements finish.
set -euo pipefail
stage=/home/fairfieldt/trashcan-exp/tp4-offline-20261007/fan-interval-20261007
old=02a62a95ab37e39c21fc3b1efffc815b7f52916fb9d0af00cc4b55e721e6ec50
new=b175a2bb41002a52a133a1d6ed7eb2988a421a4d2bac9a8005efed3fd6b0b3c3
unit=ad1d7e35f7bf8e0053f10329ea008ff842a8ff748f0d18ed7ea4a775b9451de8
printf '%s  %s\n' "$old" /usr/local/sbin/fancurve.sh | sha256sum -c -
printf '%s  %s\n' "$new" "$stage/fancurve.sh" | sha256sum -c -
printf '%s  %s\n' "$unit" /etc/systemd/system/fancurve.service | sha256sum -c -
mkdir "$stage/backup"
sudo -n cp -a /usr/local/sbin/fancurve.sh "$stage/backup/fancurve.sh"
sudo -n install -m 755 -o root -g root "$stage/fancurve.sh" /usr/local/sbin/fancurve.sh.interval-tmp
sudo -n mv /usr/local/sbin/fancurve.sh.interval-tmp /usr/local/sbin/fancurve.sh
if ! sudo -n systemctl restart fancurve.service || ! systemctl is-active --quiet fancurve.service; then
    sudo -n cp -a "$stage/backup/fancurve.sh" /usr/local/sbin/fancurve.sh
    sudo -n systemctl restart fancurve.service
    echo 'interval guard deployment failed; previous maximum-fan controller restored' >&2
    exit 1
fi
systemctl is-enabled fancurve.service
systemctl show fancurve.service -p ActiveState -p SubState -p NRestarts -p ExecMainStatus
sha256sum /usr/local/sbin/fancurve.sh /etc/systemd/system/fancurve.service
