#!/bin/bash
# Hand a Thunderbolt NHI from Linux's thunderbolt driver to vfio-pci for tbd, or give it back (run as root).
# - Its reset methods are cleared, so opening it through VFIO resets nothing (a bus reset would take the
#   controller's PCIe side down under the router).
# - vfio-pci's disable_idle_d3=1 (kernel command line) keeps it in D0 while nobody has it open.
# - The Mac Pro 6,1 has no interrupt remapping, so VFIO needs allow_unsafe_interrupts (driver/docs/vfio.md).
# - The kernel's services on the domain (thunderbolt_net, thunderbolt_stream) are unbound first: unbinding the NHI
#   under a running thunderbolt_net leaves its rings and interface behind ("TX ring 2 is still active"), and that
#   interface then writes ring registers of an NHI tbd owns. That ended once in a silent reset of the
#   machine.
# usage: vfio.sh take|give 0000:16:00.0
set -euo pipefail
cmd=$1 nhi=$2
dev=/sys/bus/pci/devices/$nhi
case $cmd in
take)
    modprobe vfio-pci
    grep -qx Y /sys/module/vfio_pci/parameters/disable_idle_d3 || { echo "boot with vfio_pci.disable_idle_d3=1" >&2; exit 1; }
    echo 1 > /sys/module/vfio_iommu_type1/parameters/allow_unsafe_interrupts
    echo > $dev/reset_method
    for d in /sys/bus/thunderbolt/devices/domain*; do
        [ "$(basename "$(readlink -f "$d/..")")" = "$nhi" ] || continue
        n=${d##*domain}
        for svc in /sys/bus/thunderbolt/devices/$n-*.*; do
            [ -e "$svc/driver" ] || continue
            echo "unbinding $(basename "$svc") from $(basename "$(readlink "$svc/driver")")"
            echo "$(basename "$svc")" > "$svc/driver/unbind"
        done
    done
    [ -e $dev/driver ] && echo $nhi > $dev/driver/unbind
    echo vfio-pci > $dev/driver_override
    echo $nhi > /sys/bus/pci/drivers_probe
    ;;
give)
    [ -e $dev/driver ] && echo $nhi > $dev/driver/unbind
    echo > $dev/driver_override
    echo default > $dev/reset_method
    echo $nhi > /sys/bus/pci/drivers_probe
    ;;
*)
    echo "usage: $0 take|give NHI" >&2
    exit 2
    ;;
esac
group=$(basename "$(readlink $dev/iommu_group)")
echo "$nhi: driver $(basename "$(readlink $dev/driver 2>/dev/null)" 2>/dev/null || echo none), power $(cat $dev/power_state), reset_method [$(cat $dev/reset_method)], IOMMU group $group:"
for d in /sys/kernel/iommu_groups/$group/devices/*; do
    echo "  $(basename $d) $(basename "$(readlink $d/driver 2>/dev/null)" 2>/dev/null || echo none)"
done
