#!/bin/sh
# Writes only disposable disk images under dross/target/disktest.
set -eu
cd "$(dirname "$0")/.."
DISK_PREPARE_ONLY=1 ./linux/disk-test.sh
debugfs=$(command -v debugfs || echo /opt/homebrew/opt/e2fsprogs/sbin/debugfs)
"$debugfs" -w -R 'write /dev/null /test/large' target/disktest/fs.img >/dev/null 2>&1
"$debugfs" -w -R 'set_inode_field /test/large size 5368709243' target/disktest/fs.img >/dev/null 2>&1
python3 - <<'PYIMAGE'
import shutil
with open('target/disktest/fs.img','rb') as source, open('target/disktest/disk.img','r+b') as image:
    image.seek(2048*512)
    shutil.copyfileobj(source,image)
PYIMAGE
zig cc -target x86_64-linux-musl -static -O2 -UNDEBUG -pthread linux/disk-write-test.c -o target/linux/fswrite-test
disk_image=target/disktest/disk.img
if [ "${BLOCK_RANGE_TEST:-0}" = 1 ]; then
    python3 - <<'PYRANGE'
import hashlib,json,os
prefix=os.urandom(1<<20);suffix=os.urandom(1<<20)
with open('target/disktest/range.img','wb') as out,open('target/disktest/fs.img','rb') as source:
    out.write(prefix)
    import shutil
    shutil.copyfileobj(source,out)
    out.write(suffix)
with open('target/disktest/range-sentinels.json','w') as out:
    json.dump([hashlib.sha256(prefix).hexdigest(),hashlib.sha256(suffix).hexdigest()],out)
PYRANGE
    cargo build -p fsd --bin block-boundary-test --target x86_64-unknown-none --release
    cp target/x86_64-unknown-none/release/block-boundary-test target/linux/block-boundary-test
    disk_image=target/disktest/range.img
fi
for phase in write verify; do
    if [ "${BLOCK_RANGE_TEST:-0}" = 1 ]; then
        cat > target/disktest/init.conf <<CONF
channel raw
channel disk
channel bounds
mount root /disk
native usbd @raw=block %block-start=2048 %block-count=131072
native block-boundary-test @raw=block @disk=service @bounds=ready %blocks=131072
wait bounds 10000
native fsd @disk=block @root=service %writable=1
check busybox 0 sh /disk/test/$phase.sh
CONF
    else
        cat > target/disktest/init.conf <<CONF
channel disk
mount root /disk
native usbd @disk=block
native fsd @disk=block @root=service %writable=1
check busybox 0 sh /disk/test/$phase.sh
CONF
    fi
    if [ "$phase" = write ]; then echo 'check fswrite-test 0' >> target/disktest/init.conf;
    else echo 'check fswrite-test 0 verify' >> target/disktest/init.conf; fi
    echo reboot >> target/disktest/init.conf
    KERNEL_FEATURES=unix INIT_CONF=target/disktest/init.conf ./run-qemu.sh \
      -drive if=none,id=stick,format=raw,file="$disk_image" -device usb-storage,bus=xhci.0,drive=stick \
      > "target/disktest/$phase.log" 2>&1
    tail -5 "target/disktest/$phase.log"
done

if [ "${BLOCK_RANGE_TEST:-0}" = 1 ]; then
    python3 - <<'PYCHECK'
import hashlib,json
with open('target/disktest/range.img','rb') as image:
    prefix=image.read(1<<20)
    image.seek(-(1<<20),2)
    suffix=image.read(1<<20)
assert [hashlib.sha256(prefix).hexdigest(),hashlib.sha256(suffix).hexdigest()]==json.load(open('target/disktest/range-sentinels.json'))
print('outside block view: unchanged')
PYCHECK
fi
