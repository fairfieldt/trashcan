#!/bin/bash
# Unload the Thunderbolt stack, reset each NHI through the kernel's PCI reset (sysfs "reset", a bus
# reset of the NHI's own link with config saved and restored), and load the given
# thunderbolt.ko/thunderbolt_net.ko/thunderbolt_stream.ko (default: the installed modules). This
# recovers a controller that stopped answering config reads. Never remove the devices or reset the
# PLX ports by hand: that crashed the PCI core (pcie_update_aspm_capable oops, 2026-10-06).
# usage: tb-reset.sh [MODDIR]
set -u
moddir=${1:-}
NS="sudo ip netns exec tbB"
for x in $($NS ls /sys/class/net 2>/dev/null); do [ "$x" = lo ] || $NS ip link set "$x" netns 1; done
sudo rmdir /sys/kernel/config/thunderbolt/stream/*/* /sys/kernel/config/thunderbolt/stream/* 2>/dev/null
sudo rmmod thunderbolt_ibverbs thunderbolt_stream thunderbolt_dma_test thunderbolt_net 2>/dev/null
sudo rmmod thunderbolt || { echo "rmmod thunderbolt failed" >&2; exit 1; }
for nhi in 16:00.0 5d:00.0 a4:00.0; do sudo sh -c "echo 1 > /sys/bus/pci/devices/0000:$nhi/reset"; done
if [ -n "$moddir" ]; then
  sudo insmod "$moddir/thunderbolt.ko" &&
  sudo insmod "$moddir/thunderbolt_net.ko" throttling=${THROTTLING:-128000} &&
  sudo insmod "$moddir/thunderbolt_stream.ko" || exit 1
else
  sudo modprobe thunderbolt_net && sudo modprobe thunderbolt_stream
fi
sleep 10
ls /sys/bus/thunderbolt/devices | tr '\n' ' '; echo
