#!/bin/bash
# Installs applesmc with patch 0009 (the port lights) on Mac Pros as a module override for their running kernel,
# without a reboot, until a kernel with 0009 built in is rolled out.
# Builds it in the kbuild VM against the host's installed headers. Their host tools are a mix of arm64 and x86-64
# binaries, so they are swapped for the arm64 ones of the VM's 7.3 tree. Installs it as
# /lib/modules/KREL/updates/applesmc.ko and reloads applesmc. fancurve stops for those few seconds, so the SMC runs
# the fan meanwhile.
# Undo on a host: sudo rm /lib/modules/$(uname -r)/updates/applesmc.ko; sudo depmod; then reboot or reload applesmc.
# usage (on the Mac): kernel/applesmc-override.sh HOST...
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
vm=kbuild
stage=$HOME/dev/claude-work/trashcan-kbuild/applesmc-override
patch=$here/patches-7.3/0009-applesmc-mac-pro-port-lights.patch
[ $# -gt 0 ] || { echo "usage: $0 HOST..." >&2; exit 2; }

limactl list --format '{{.Status}}' "$vm" | grep -q Running || limactl start "$vm"
vmhome=$(limactl shell "$vm" -- sh -c 'echo $HOME')

for h in "$@"; do
    krel=$(ssh "$h" uname -r)
    case $krel in
        7.3.0-rc7-*) ver=7.3-rc7 ;;
        *) echo "$h: kernel $krel: only 7.3.0-rc7 builds are handled" >&2; exit 1 ;;
    esac
    w=applesmc-override/$krel
    mkdir -p "$stage/$krel"
    ko=$stage/$krel/applesmc.ko
    if [ ! -e "$ko" ] || [ "$patch" -nt "$ko" ]; then
        echo "building for $krel"
        limactl shell "$vm" -- bash -c "rm -rf ~/$w && mkdir -p ~/$w/hdr ~/$w/src"
        ssh "$h" "tar -C /usr/src -c linux-headers-$krel" | limactl shell "$vm" -- tar -C "$vmhome/$w/hdr" -x
        limactl copy "$patch" "$vm:$vmhome/$w/0009.patch"
        limactl shell "$vm" -- bash -c "set -e; cd ~/$w; tree=~/kernel/7.3/linux-$ver; h=hdr/linux-headers-$krel
            for f in \$(cd \$h && find scripts tools -type f -perm -u+x); do
                if file -b \$h/\$f | grep -q '^ELF' && [ -x \$tree/\$f ]; then cp \$tree/\$f \$h/\$f; fi
            done
            tar -xzf ~/kernel/7.3/linux-$ver.tar.gz -C src --strip-components=1 linux-$ver/drivers/hwmon/applesmc.c
            patch -s -d src -p1 < 0009.patch
            cp src/drivers/hwmon/applesmc.c . && echo 'obj-m := applesmc.o' > Makefile
            make -s -C \$PWD/\$h M=\$PWD ARCH=x86_64 CROSS_COMPILE=x86_64-linux-gnu- modules"
        limactl copy "$vm:$vmhome/$w/applesmc.ko" "$ko"
    fi
    scp -q "$ko" "$h":applesmc-override.ko
    ssh "$h" "set -e; k=\$(uname -r)
        sudo install -D -m 0644 ~/applesmc-override.ko /lib/modules/\$k/updates/applesmc.ko; rm ~/applesmc-override.ko
        sudo depmod \$k
        fan=\$(systemctl is-active fancurve || true)
        [ \"\$fan\" = active ] && sudo systemctl stop fancurve
        sudo rmmod applesmc 2>/dev/null || true
        sudo modprobe applesmc
        [ \"\$fan\" = active ] && sudo systemctl start fancurve
        echo \"$h: \$(ls /sys/class/leds | grep -c '^smc::') port lights, fancurve \$(systemctl is-active fancurve)\""
done
