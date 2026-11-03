#!/bin/sh
# Exercise the exact empty-image/scoped-view hardware configuration in QEMU.
set -eu
cd "$(dirname "$0")/.."
d=target/hardware-fs-test
mkdir -p "$d"
mkfs=$(command -v mkfs.ext4 || echo /opt/homebrew/opt/e2fsprogs/sbin/mkfs.ext4)
"$mkfs" -q -F -b 4096 -O ^orphan_file,^metadata_csum_seed "$d/fs.img" 64M
uuid=$(python3 - "$d" <<'PYIMAGE'
import hashlib,json,os,sys,uuid
from pathlib import Path
p=Path(sys.argv[1]);fs=(p/'fs.img').read_bytes();prefix=os.urandom(1<<20);suffix=os.urandom(1<<20)
(p/'disk.img').write_bytes(prefix+fs+suffix)
(p/'sentinels.json').write_text(json.dumps([hashlib.sha256(prefix).hexdigest(),hashlib.sha256(suffix).hexdigest()]))
print(uuid.UUID(bytes=fs[1024+104:1024+120]))
PYIMAGE
)
zig cc -target x86_64-linux-musl -static -O2 -UNDEBUG -pthread linux/disk-write-test.c -o target/linux/fswrite-test
for phase in write verify rewrite verify-again reject; do
    expected=$uuid
    args=hardware-$phase
    [ "$phase" != verify-again ] || args=hardware-verify
    if [ "$phase" = reject ]; then
        expected=00000000-0000-0000-0000-000000000000
        args=expect-unavailable
        python3 - "$d/disk.img" "$d/before-reject.sha" <<'PYSHA'
import hashlib,sys
open(sys.argv[2],'w').write(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())
PYSHA
    fi
    cat > "$d/init.conf" <<CONF
channel disk
mount root /disk
native usbd @disk=block %block-start=2048 %block-count=131072
native fsd @disk=block @root=service %writable=1 %uuid=$expected
check fswrite-test 0 $args
reboot
CONF
    KERNEL_FEATURES=unix INIT_CONF="$d/init.conf" ./run-qemu.sh \
      -drive if=none,id=stick,format=raw,file="$d/disk.img" -device usb-storage,bus=xhci.0,drive=stick \
      > "$d/$phase.log" 2>&1
    tail -5 "$d/$phase.log"
    # Copy only the delegated filesystem, replay with Linux tools first, and
    # reject repair diagnostics even when e2fsck exits zero.
    python3 - "$d/disk.img" "$d/linux-replay-$phase.img" <<'PYCOPY'
import sys
from pathlib import Path
Path(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes()[1<<20:-(1<<20)])
PYCOPY
    python3 user/fsd/tests/check-recovery.py "$d/linux-replay-$phase.img" > "$d/linux-replay-$phase.log" 2>&1

done
python3 - "$d" <<'PYCHECK'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);data=(p/'disk.img').read_bytes()
assert [hashlib.sha256(data[:1<<20]).hexdigest(),hashlib.sha256(data[-(1<<20):]).hexdigest()]==json.loads((p/'sentinels.json').read_text())
assert hashlib.sha256(data).hexdigest()==(p/'before-reject.sha').read_text()
print('hardware config: persisted; outside view unchanged; UUID mismatch made no disk writes')
PYCHECK
