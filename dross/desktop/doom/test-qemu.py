#!/usr/bin/env python3
"""Build and run an isolated native-window Doom test image. Never alters normal ESP."""
import os,shutil,socket,subprocess,tarfile,time
from pathlib import Path
here=Path(__file__).resolve().parent
workspace=here.parent
dross=workspace.parent
out=workspace/'target/doom-qemu'
profile='release'
def run(args,**kwargs):subprocess.run([str(a) for a in args],check=True,**kwargs)
run([here/'build-musl.sh'],env={**os.environ,"DOOM_FEATURES":"acceptance"})
run(['cargo','build','--manifest-path',workspace/'Cargo.toml','-p','oscar-desktop-native','--features','native','--bins','--release','--target','x86_64-unknown-none'],cwd=dross)
kernel=out/'kernel'
run(['cargo','build','-p','dross','--release','--target','x86_64-unknown-none','--target-dir',kernel],cwd=dross)
run(['cargo','build','-p','dross-boot','--release','--target','x86_64-unknown-uefi','--target-dir',kernel],cwd=dross)
esp=out/'esp'
if esp.exists():shutil.rmtree(esp)
(esp/'EFI/BOOT').mkdir(parents=True)
modules=esp/'modules';modules.mkdir()
shutil.copy2(kernel/'x86_64-unknown-none/release/dross',esp/'dross.elf')
shutil.copy2(kernel/'x86_64-unknown-uefi/release/dross-boot.efi',esp/'EFI/BOOT/BOOTX64.EFI')
native=workspace/'target/x86_64-unknown-none/release'
musl=workspace/'target/x86_64-unknown-linux-musl/release'
for name,source in [('init',native/'oscar-desktop-test-init'),('oscar-windowd',native/'oscar-windowd'),('oscar-doom',musl/'oscar-doom'),('oscar-windowcheck',musl/'oscar-windowcheck')]:shutil.copy2(source,modules/name)
with tarfile.open(modules/'rootfs.tar','w',format=tarfile.USTAR_FORMAT) as tar:
    for name in ['freedoom1.wad','COPYING.txt','CREDITS.txt','CREDITS-MUSIC.txt']:
        tar.add(workspace/'target/freedoom'/name,arcname=name)
firmware=None
for directory in ['/opt/homebrew/share/qemu','/usr/share/qemu','/usr/share/OVMF']:
    directory=Path(directory)
    for code,variables in [('edk2-x86_64-code.fd','edk2-i386-vars.fd'),('OVMF_CODE.fd','OVMF_VARS.fd')]:
        if (directory/code).exists():firmware=directory/code;shutil.copy2(directory/variables,out/'vars.fd');break
    if firmware:break
if not firmware:raise SystemExit('OVMF not found')
cmd=['qemu-system-x86_64','-machine','q35','-accel','tcg,thread=multi','-cpu','IvyBridge,+x2apic','-smp',os.getenv('SMP','4'),'-m','2G','-device','intel-iommu,intremap=off','-drive',f'if=pflash,format=raw,readonly=on,file={firmware}','-drive',f'if=pflash,format=raw,file={out}/vars.fd','-drive',f'format=raw,file=fat:rw:{esp}','-device','isa-debug-exit,iobase=0xf4,iosize=4','-monitor',f'unix:{out}/monitor.sock,server,nowait','-serial','stdio','-display','none','-no-reboot','-net','none']
monitor=out/'monitor.sock'
if monitor.exists():monitor.unlink()
with (out/'qemu.log').open('w') as log:
    process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    deadline=time.monotonic()+120
    captured=False
    while process.poll() is None:
        if time.monotonic()>deadline:
            process.kill();process.wait();raise SystemExit(f'QEMU timed out; see {out}/qemu.log')
        if not captured and 'windowcheck: capture' in (out/'qemu.log').read_text(errors='replace'):
            with socket.socket(socket.AF_UNIX) as connection:
                connection.connect(str(monitor));connection.sendall(f'screendump {out}/oscar-doom.ppm\n'.encode());time.sleep(.1)
            captured=True
        time.sleep(.02)
    status=process.returncode
text=(out/'qemu.log').read_text(errors='replace')
shutil.copy2(out/'qemu.log',out/f'qemu-smp{os.getenv("SMP","4")}.log')
print(text[text.find('init started'):])
if status!=33 or 'desktop test: ok' not in text:raise SystemExit('native Doom/window test failed')
