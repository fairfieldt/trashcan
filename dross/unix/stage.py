#!/usr/bin/env python3
"""Stage only this suite's ESP; never touch dross/target/esp or a mounted ESP."""
from pathlib import Path
import shutil, tarfile, os, io
here=Path(__file__).resolve().parent; dross=here.parent; out=here/'target';esp=out/'esp'
(esp/'EFI/BOOT').mkdir(parents=True,exist_ok=True);(esp/'modules').mkdir(exist_ok=True)
shutil.copy2(out/'kernel/x86_64-unknown-uefi/release/dross-boot.efi',esp/'EFI/BOOT/BOOTX64.EFI')
shutil.copy2(out/'kernel/x86_64-unknown-none/release/dross',esp/'dross.elf')
shutil.copy2(out/'native/x86_64-unknown-none/release/unix-init',esp/'modules/init')
shutil.copy2(out/'unix-probe',esp/'modules/unix-probe')
shutil.copy2(out/'native/x86_64-unknown-linux-musl/release/unix-terminal-bridge',esp/'modules/unix-terminal-bridge')
(esp/'modules/entropy.seed').write_bytes(os.urandom(32))
shutil.copy2(out/'native/x86_64-unknown-none/release/unix-netd',esp/'modules/unix-netd')
shutil.copy2(out/'native/x86_64-unknown-none/release/unix-nic',esp/'modules/unix-nic')
shutil.copy2(out/'kernel/x86_64-unknown-none/release/pcid',esp/'modules/pcid')
busybox=out/'src/busybox-1.37.0/busybox_unstripped'
with tarfile.open(esp/'modules/rootfs.tar','w',format=tarfile.USTAR_FORMAT) as tar:
 def data(name,content,mode=0o644):
  info=tarfile.TarInfo(name);info.size=len(content);info.mode=mode;tar.addfile(info,io.BytesIO(content))
 data('bin/busybox',busybox.read_bytes(),0o755)
 data('bin/unix-probe',(out/'unix-probe').read_bytes(),0o755)
 data('usr/share/licenses/busybox/LICENSE',(out/'src/busybox-1.37.0/LICENSE').read_bytes())
 data('usr/share/licenses/dropbear/LICENSE',(out/'src/dropbear-2026.94/LICENSE').read_bytes())
 for binary in ['dropbear','dropbearkey','dropbearconvert']:
  data('usr/sbin/'+binary,(out/'src/dropbear-2026.94'/binary).read_bytes(),0o755)
 data('etc/test-hostkey',(out/'test-hostkey').read_bytes(),0o600)
 data('root/.ssh/authorized_keys',(out/'test-client.pub').read_bytes(),0o600)
 for path in (out/'src/busybox-1.37.0/busybox.links').read_text().splitlines():
  path = path.lstrip('/')
  info=tarfile.TarInfo(path);info.type=tarfile.SYMTYPE;info.linkname='/bin/busybox';info.mode=0o777;tar.addfile(info)
 # Dynamically linked programs: musl's dynamic linker and BusyBox from Alpine, and linux/dyntest.
 with tarfile.open(here/'vendor-cache/alpine-minirootfs-3.24.3-x86_64.tar.gz') as alpine:
  data('lib/ld-musl-x86_64.so.1',alpine.extractfile('./lib/ld-musl-x86_64.so.1').read(),0o755)
  data('opt/alpine/bin/busybox',alpine.extractfile('./bin/busybox').read(),0o755)
 data('usr/share/licenses/musl/COPYRIGHT',(out/'musl-COPYRIGHT').read_bytes())
 data('usr/bin/dyntest',(out/'dyntest').read_bytes(),0o755)
 data('usr/lib/libdrosstest.so',(out/'libdrosstest.so').read_bytes(),0o755)
 data('etc/passwd',b'root:x:0:0:root:/root:/bin/sh\n')
 data('etc/group',b'root:x:0:\n')
 data('etc/shells',b'/bin/sh\n/bin/ash\n')
 data('etc/os-release',b'NAME=Oscar\nID=oscar\n')
print(esp)
