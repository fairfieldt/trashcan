#!/bin/bash
# On a Mac Pro: install a kernel's image and headers from ~/kernel/7.3/debs (never linux-libc-dev) and either boot it
# once with grub-reboot (GRUB_DEFAULT unchanged, so a power cycle returns to it) or make it GRUB's default. Does not
# reboot.
# usage: kernel-boot.sh once|default KREL     (KREL like 7.3.0-rc7-p2p; DEBS overrides the package directory)
set -euo pipefail
mode=$1 krel=$2
debs=${DEBS:-$HOME/kernel/7.3/debs}
img=$(ls -t "$debs"/linux-image-"${krel}"_*_amd64.deb | head -1)
hdr=$(ls -t "$debs"/linux-headers-"${krel}"_*_amd64.deb | head -1)
want=$(dpkg-deb -f "$img" Version)
have=$(dpkg-query -W -f '${Version} ${Status},' "linux-image-$krel" "linux-headers-$krel" 2>/dev/null || true)
# --force-unsafe-io: no fsync per file; an original Apple SSD takes up to seconds per cache flush (nodes/macpro/README.md).
[ "$have" = "$want install ok installed,$want install ok installed," ] || sudo -n dpkg --force-unsafe-io -i "$img" "$hdr"
cfg=$(sudo -n cat /boot/grub/grub.cfg)    # not world-readable on every node
sub=$(echo "$cfg" | grep -o "submenu 'Advanced options[^{]*\$menuentry_id_option '[^']*'" | sed "s/.*option '//; s/'\$//")
ent=$(echo "$cfg" | grep -o "menuentry '[^']*, with Linux $krel'[^{]*\$menuentry_id_option '[^']*'" |
    sed "s/.*option '//; s/'\$//")
[ -n "$sub" ] && [ -n "$ent" ] || { echo "no GRUB entry for $krel" >&2; exit 1; }
case $mode in
    once) sudo -n grub-reboot "$sub>$ent"; sudo -n grub-editenv list ;;
    default)
        sudo -n sed -i "s#^GRUB_DEFAULT=.*#GRUB_DEFAULT=\"$sub>$ent\"#" /etc/default/grub
        sudo -n update-grub 2>/dev/null ;;
    *) echo "mode must be once or default" >&2; exit 1 ;;
esac
grep "^GRUB_DEFAULT=" /etc/default/grub
echo "installed $(dpkg-query -W -f '${Version}' "linux-image-$krel"); running $(uname -r)"
