#!/usr/bin/env python3
"""Stages the Unix userland for a hardware boot: writes ../target/linux/rootfs.tar (replacing build-linux.sh's) with
the files under linux/rootfs, optional firmware, BusyBox (with its applet links), Dropbear, and an /etc/rc that
starts sshd; and copies unix-netd and unix-terminal-bridge (a shell on a PTY for the desktop's terminal) next to
the other modules. Keys come from files, so nothing machine-specific is
in the repository.

usage: unix/stage-hw.py --host-key OPENSSH_PRIVATE_KEY --authorized-keys FILE [--firmware DIR]
Run after ./build-linux.sh and unix/build.sh; then build.sh with KERNEL_FEATURES=unix.
"""
import argparse, io, shutil, tarfile
from pathlib import Path

here = Path(__file__).resolve().parent
dross = here.parent
out = here / 'target'
ap = argparse.ArgumentParser()
ap.add_argument('--host-key', required=True)
ap.add_argument('--authorized-keys', required=True)
ap.add_argument('--firmware')
args = ap.parse_args()

linux = dross / 'target/linux'
shutil.copy2(out / 'native/x86_64-unknown-none/release/unix-netd', linux / 'unix-netd')
shutil.copy2(out / 'native/x86_64-unknown-linux-musl/release/unix-terminal-bridge', linux / 'unix-terminal-bridge')
busybox = out / 'src/busybox-1.37.0'
# Also a module, so init.conf can start `busybox sh /etc/rc`.
shutil.copy2(busybox / 'busybox_unstripped', linux / 'busybox')
dropbear = out / 'src/dropbear-2026.94'
rc = b'''#!/bin/sh
mkdir -p /etc/dropbear
dropbearconvert openssh dropbear /etc/hostkey.pem /etc/dropbear/hostkey
exec dropbear -F -r /etc/dropbear/hostkey -p 22
'''
with tarfile.open(linux / 'rootfs.tar', 'w', format=tarfile.USTAR_FORMAT) as tar:
    def data(name, content, mode=0o644):
        info = tarfile.TarInfo(name)
        info.size = len(content)
        info.mode = mode
        tar.addfile(info, io.BytesIO(content))
    for f in sorted((dross / 'linux/rootfs').rglob('*')):
        if f.is_file():
            data(str(f.relative_to(dross / 'linux/rootfs')), f.read_bytes())
    if args.firmware:
        for f in sorted(Path(args.firmware).rglob('*')):
            if f.is_file():
                data('lib/firmware/' + str(f.relative_to(args.firmware)), f.read_bytes())
    data('bin/busybox', (busybox / 'busybox_unstripped').read_bytes(), 0o755)
    for path in (busybox / 'busybox.links').read_text().splitlines():
        info = tarfile.TarInfo(path.lstrip('/'))
        info.type = tarfile.SYMTYPE
        info.linkname = '/bin/busybox'
        info.mode = 0o777
        tar.addfile(info)
    for binary in ['dropbear', 'dropbearkey', 'dropbearconvert']:
        data('usr/sbin/' + binary, (dropbear / binary).read_bytes(), 0o755)
    data('usr/share/licenses/busybox/LICENSE', (busybox / 'LICENSE').read_bytes())
    data('usr/share/licenses/dropbear/LICENSE', (dropbear / 'LICENSE').read_bytes())
    data('etc/rc', rc, 0o755)
    data('etc/hostkey.pem', Path(args.host_key).read_bytes(), 0o600)
    data('root/.ssh/authorized_keys', Path(args.authorized_keys).read_bytes(), 0o600)
    data('etc/passwd', b'root:x:0:0:root:/root:/bin/sh\n')
    data('etc/group', b'root:x:0:\n')
    data('etc/shells', b'/bin/sh\n/bin/ash\n')
    data('etc/os-release', b'NAME=Oscar\nID=oscar\n')
print(linux / 'rootfs.tar')
