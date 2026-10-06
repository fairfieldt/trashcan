#!/bin/bash
# P2P evidence run; idle suspend blocked for its duration (GDM greeter suspends after 900 s idle).
out=$(cd "$(dirname "$0")" && pwd); cd ~/trashcan/bench/xgpu
{ uname -a; cat /proc/cmdline; grep -m1 "model name" /proc/cpuinfo; grep -E "^CONFIG_(PCI_P2PDMA|DMABUF_MOVE_NOTIFY|INTEL_IOMMU|IOMMU_DEFAULT)" /boot/config-$(uname -r); sudo dmesg | grep -E "iommu: Default domain"; for d in 02:00.0 06:00.0; do echo "$d iommu_group $(basename $(readlink /sys/bus/pci/devices/0000:$d/iommu_group))"; done; } > $out/system.txt 2>&1
lspci -nn -s 00:00.0 > $out/lspci.txt; lspci -nn -s 00:02.0 >> $out/lspci.txt; lspci -nn -s 00:03.0 >> $out/lspci.txt
sudo dmesg > $out/dmesg-before.txt
( while :; do echo "$(date +%s.%N) $(cat /sys/class/drm/card0/device/hwmon/hwmon*/freq1_input) $(cat /sys/class/drm/card0/device/hwmon/hwmon*/freq2_input) $(cat /sys/class/drm/card1/device/hwmon/hwmon*/freq1_input) $(cat /sys/class/drm/card1/device/hwmon/hwmon*/freq2_input)"; sleep 0.5; done > $out/clocks.txt 2>&1 < /dev/null ) & cpid=$!
for i in 1 2 3 4 5; do ./xgpu p2p-verify > $out/xgpu-verify-$i.txt 2>&1; echo "exit $?" >> $out/xgpu-verify-$i.txt; done
./xgpu p2p > $out/xgpu-p2p.txt 2>&1; echo "exit $?" >> $out/xgpu-p2p.txt
kill $cpid
sudo dmesg > $out/dmesg-after.txt
