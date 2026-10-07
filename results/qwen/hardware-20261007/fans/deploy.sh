#!/bin/bash
set -euo pipefail
stage=/home/fairfieldt/trashcan-exp/tp4-offline-20261007/fan-max-20261007
cd "$stage"
mkdir backup
systemctl show fancurve.service -p ActiveState -p SubState -p UnitFileState -p NRestarts > backup/service-before.txt
sudo -n cp -p /usr/local/sbin/fancurve.sh backup/fancurve.sh
sudo -n cp -p /etc/systemd/system/fancurve.service backup/fancurve.service
sha256sum fancurve.sh fancurve.service backup/fancurve.sh backup/fancurve.service
sudo -n install -m 755 fancurve.sh /usr/local/sbin/fancurve.sh
sudo -n install -m 644 fancurve.service /etc/systemd/system/fancurve.service
sudo -n systemctl daemon-reload
sudo -n systemctl enable fancurve.service
sudo -n systemctl restart fancurve.service
systemctl show fancurve.service -p ActiveState -p SubState -p UnitFileState -p NRestarts -p ExecMainStatus
sha256sum /usr/local/sbin/fancurve.sh /etc/systemd/system/fancurve.service
