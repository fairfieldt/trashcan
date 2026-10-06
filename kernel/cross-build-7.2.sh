#!/bin/bash
# Cross-builds the 7.2 kernel packages on this Mac in the Lima VM "kbuild" (Debian 13 arm64,
# gcc-x86-64-linux-gnu 14.2.0-19, the same GCC as incite) and copies the .debs to incite:~/kernel/7.2/debs/.
# The kernel tree lives on the VM's own ext4 disk (macOS filesystems are case-insensitive).
# One-time VM setup:
#   limactl create --name=kbuild --cpus=30 --memory=64 --disk=200 --vm-type=vz --mount-none \
#       --tty=false template://debian-13
#   limactl start kbuild
#   limactl shell kbuild -- sudo dpkg --add-architecture amd64   # linux-headers rebuilds its host tools for amd64
#   limactl shell kbuild -- sudo apt-get install -y --no-install-recommends build-essential \
#       gcc-x86-64-linux-gnu binutils-x86-64-linux-gnu bc bison flex libssl-dev libelf-dev libdw-dev \
#       dwarves rsync kmod cpio python3 dpkg-dev debhelper git curl ca-certificates xz-utils patch \
#       libssl-dev:amd64 libc6-dev:amd64 libelf-dev:amd64 zlib1g-dev:amd64
# usage (on the Mac): [REV=N] [JOBS=N] kernel/cross-build-7.2.sh   (lower JOBS when the Mac is busy)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
vm=kbuild
stage=$HOME/dev/claude-work/trashcan-kbuild
mkdir -p "$stage"

limactl list --format '{{.Status}}' "$vm" | grep -q Running || limactl start "$vm"
vmhome=$(limactl shell "$vm" -- sh -c 'echo $HOME')
limactl shell "$vm" -- rm -rf trashcan-kernel
limactl shell "$vm" -- mkdir -p trashcan-kernel kernel/7.2
limactl copy -r "$here/build-7.2.sh" "$here/patches-7.2" "$here/configs" "$vm:$vmhome/trashcan-kernel/"
limactl shell "$vm" -- bash -c "REV=${REV:-2} JOBS=${JOBS:-} ~/trashcan-kernel/build-7.2.sh" 2>&1 | tee "$stage/build.log"
rm -f "$stage"/*.deb
for f in $(limactl shell "$vm" -- sh -c 'ls ~/kernel/7.2/linux-image-*.deb ~/kernel/7.2/linux-headers-*.deb'); do
    limactl copy "$vm:$f" "$stage/"
done
ls -la "$stage"/*.deb
ssh incite 'rm -rf ~/kernel/7.2/debs && mkdir -p ~/kernel/7.2/debs'
scp -q "$stage"/*.deb incite:kernel/7.2/debs/
echo "copied to incite:~/kernel/7.2/debs/"
