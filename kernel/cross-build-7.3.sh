#!/bin/bash
# Cross-builds the 7.3 kernel packages on this Mac in the Lima VM "kbuild" (Debian 13 arm64, gcc-x86-64-linux-gnu
# 14.2.0-19, the same GCC as the Mac Pros); one-time VM setup is in cross-build-7.2.sh. The tree lives on the VM's
# own ext4 disk (macOS filesystems are case-insensitive).
# Leaves the image and headers .debs and the resulting config in ~/dev/claude-work/trashcan-kbuild/ (the 7.2.9
# packages there stay) and, if HOSTS is set, copies the .debs to each host's ~/kernel/7.3/debs/.
# usage (on the Mac): [VER=7.3-rc7] [REV=N] [JOBS=N] [KLOCAL=-p2p-x] [DIAG=1] [HOSTS="inflight ..."] kernel/cross-build-7.3.sh
#   (lower JOBS when the Mac is busy; DIAG=1 adds patches-7.3-diag/ and needs its own KLOCAL, see build-7.3.sh)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
vm=kbuild
stage=$HOME/dev/claude-work/trashcan-kbuild
mkdir -p "$stage"

limactl list --format '{{.Status}}' "$vm" | grep -q Running || limactl start "$vm"
vmhome=$(limactl shell "$vm" -- sh -c 'echo $HOME')
limactl shell "$vm" -- rm -rf trashcan-kernel-7.3
limactl shell "$vm" -- mkdir -p trashcan-kernel-7.3 kernel/7.3
limactl copy -r "$here/build-7.3.sh" "$here/patches-7.3" "$here/patches-7.3-diag" "$here/configs" "$vm:$vmhome/trashcan-kernel-7.3/"
limactl shell "$vm" -- bash -c "VER=${VER:-} REV=${REV:-} JOBS=${JOBS:-} SHA256=${SHA256:-} KLOCAL=${KLOCAL:-} DIAG=${DIAG:-0} ~/trashcan-kernel-7.3/build-7.3.sh" \
    2>&1 | tee "$stage/build-7.3.log"
rm -f "$stage"/linux-image-7.3*.deb "$stage"/linux-headers-7.3*.deb "$stage"/config-7.3*
for f in $(limactl shell "$vm" -- sh -c 'ls ~/kernel/7.3/linux-image-*.deb ~/kernel/7.3/linux-headers-*.deb ~/kernel/7.3/config-*'); do
    limactl copy "$vm:$f" "$stage/"
done
ls -la "$stage"/linux-*-7.3*.deb "$stage"/config-7.3*
for h in ${HOSTS:-}; do
    ssh "$h" 'rm -rf ~/kernel/7.3/debs && mkdir -p ~/kernel/7.3/debs'
    scp -q "$stage"/linux-image-7.3*.deb "$stage"/linux-headers-7.3*.deb "$h":kernel/7.3/debs/
    echo "copied to $h:~/kernel/7.3/debs/"
done
