#!/bin/bash
# usage: collect.sh NAME. Records boot state, then P2P/host bandwidth and one P2P data check.
out=~/trashcan/results-p2p-boot-20261005/$1; mkdir -p $out; cd ~/trashcan/bench/xgpu
cat /proc/cmdline > $out/cmdline.txt
{ sudo dmesg | grep -E "iommu: Default domain|DMAR: IOMMU enabled|DMAR: Intel|pcie_bus|Disabling IOMMU"; for d in 02:00.0 06:00.0; do echo "$d iommu_group $(basename "$(readlink /sys/bus/pci/devices/0000:$d/iommu_group)" 2>/dev/null)"; done; } > $out/iommu.txt
for d in 00:02.0 00:03.0 02:00.0 06:00.0; do echo "== $d"; sudo lspci -vvv -s $d | grep -E "MaxPayload|LnkSta:|ACSCtl"; done > $out/pcie.txt
for c in 0 1; do echo "card$c $(cat /sys/class/drm/card$c/device/vbios_version)"; done > $out/vbios.txt
./xgpu p2p-wall > $out/p2p-wall.txt 2>&1; echo "exit $?" >> $out/p2p-wall.txt
./xgpu p2p-verify > $out/p2p-verify.txt 2>&1; echo "exit $?" >> $out/p2p-verify.txt
sudo dmesg --level=err,warn > $out/dmesg-warn.txt
grep -E "P2P|host|checked|p2p-verify:" $out/p2p-wall.txt $out/p2p-verify.txt | grep -E "65536|32768|checked|p2p-verify:"
