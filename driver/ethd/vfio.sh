#!/bin/bash
# Hand a BCM57762 port from Linux's tg3 to vfio-pci for ethd, or give it back (run as root).
# Taking a port refuses unless all of these hold, so the management port can't be taken by mistake:
# - its station address is the one given (EXPECTED_MAC);
# - its interface has no IPv4 or global IPv6 address, no route, and isn't netconsole's interface;
# - it is alone in its IOMMU group.
# Its reset methods are cleared, so opening it through VFIO resets nothing; vfio-pci's disable_idle_d3=1 (kernel
# command line) keeps it in D0. The Mac Pro 6,1 has no interrupt remapping, so VFIO needs allow_unsafe_interrupts
# (../docs/vfio.md). Giving it back binds tg3 again, which resets the chip.
# usage: vfio.sh take 0000:0c:00.0 EXPECTED_MAC | vfio.sh give 0000:0c:00.0
set -euo pipefail
cmd=$1 addr=$2
dev=/sys/bus/pci/devices/$addr
[ -e "$dev" ] || { echo "$addr: no such PCI function" >&2; exit 1; }
[ "$(cat $dev/vendor)" = 0x14e4 ] || { echo "$addr is not a Broadcom device" >&2; exit 1; }
refuse() { echo "refusing to take $addr: $*" >&2; exit 1; }
case $cmd in
take)
    want=${3:?expected MAC address}
    want=${want,,}
    group=$(basename "$(readlink $dev/iommu_group 2>/dev/null)" 2>/dev/null) || refuse "no IOMMU group (is the IOMMU on?)"
    [ "$(ls /sys/kernel/iommu_groups/$group/devices)" = "$addr" ] || refuse "IOMMU group $group has other devices"
    drv=$(basename "$(readlink $dev/driver 2>/dev/null)" 2>/dev/null || true)
    if [ "$drv" = tg3 ]; then
        ifname=$(ls $dev/net)
        mac=$(cat $dev/net/$ifname/address)
        [ "$mac" = "$want" ] || refuse "$ifname has address $mac, not $want"
        [ -z "$(ip -4 -o addr show dev $ifname)" ] || refuse "$ifname has an IPv4 address"
        [ -z "$(ip -6 -o addr show dev $ifname scope global)" ] || refuse "$ifname has a global IPv6 address"
        [ -z "$(ip route show dev $ifname; ip -6 route show dev $ifname | grep -v '^fe80::/64')" ] || refuse "$ifname has routes"
        ip route show default | grep -qw "dev $ifname" && refuse "$ifname carries the default route"
        if dmesg | grep -q "netconsole.*interface name '$ifname'"; then refuse "$ifname is netconsole's interface"; fi
        for t in /sys/kernel/config/netconsole/*/dev_name; do
            [ -e "$t" ] && [ "$(cat $t)" = "$ifname" ] && refuse "$ifname is netconsole's interface"
        done
        echo "taking $addr ($ifname, $mac) from tg3"
        echo $addr > $dev/driver/unbind
    elif [ "$drv" != vfio-pci ] && [ -n "$drv" ]; then
        refuse "bound to $drv"
    fi
    modprobe vfio-pci
    grep -qx Y /sys/module/vfio_pci/parameters/disable_idle_d3 || { echo "boot with vfio_pci.disable_idle_d3=1" >&2; exit 1; }
    echo 1 > /sys/module/vfio_iommu_type1/parameters/allow_unsafe_interrupts
    echo > $dev/reset_method
    echo vfio-pci > $dev/driver_override
    [ -e $dev/driver ] || echo $addr > /sys/bus/pci/drivers_probe
    ;;
give)
    [ -e $dev/driver ] && echo $addr > $dev/driver/unbind
    echo > $dev/driver_override
    echo default > $dev/reset_method
    echo $addr > /sys/bus/pci/drivers_probe
    sleep 1
    ;;
*)
    echo "usage: $0 take ADDRESS EXPECTED_MAC | give ADDRESS" >&2
    exit 2
    ;;
esac
echo "$addr: driver $(basename "$(readlink $dev/driver 2>/dev/null)" 2>/dev/null || echo none), power $(cat $dev/power_state), reset_method [$(cat $dev/reset_method)]"
[ -d $dev/net ] && echo "interface $(ls $dev/net)"
exit 0
