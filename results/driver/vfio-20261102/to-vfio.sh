#!/bin/bash
# Move one D700 from amdgpu to vfio-pci without resetting it (run as root under the gpuq lease).
# - The GPU's reset methods are cleared, so opening it through VFIO does no function reset.
# - Its HDMI audio function is unbound and kept driverless (driver_override "none"), so it is not in vfio-pci's
#   device set and a last close cannot trigger a bus reset; the IOMMU group stays viable with an unbound member.
# - vfio-pci's disable_idle_d3=1 (kernel command line) keeps the idle GPU in D0 (NoSoftRst-: D3hot->D0 resets it).
# usage: to-vfio.sh BUS   (e.g. 02)
set -euo pipefail
gpu=0000:$1:00.0 audio=0000:$1:00.1
dev=/sys/bus/pci/devices
modprobe vfio-pci
cat /sys/module/vfio_pci/parameters/disable_idle_d3 | grep -qx Y
echo none > $dev/$audio/driver_override
[ -e $dev/$audio/driver ] && echo $audio > $dev/$audio/driver/unbind
echo > $dev/$gpu/reset_method
[ -e $dev/$gpu/driver ] && echo $gpu > $dev/$gpu/driver/unbind
echo vfio-pci > $dev/$gpu/driver_override
echo $gpu > /sys/bus/pci/drivers_probe
readlink $dev/$gpu/driver $dev/$audio/driver || true
echo "reset_method=[$(cat $dev/$gpu/reset_method)] power=$(cat $dev/$gpu/power_state) group=$(basename $(readlink $dev/$gpu/iommu_group))"
ls -l /dev/vfio/
