#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
d=../../target/fsd-adapter
mkdir -p "$d"
mkfs=${MKFS_EXT4:-/opt/homebrew/opt/e2fsprogs/sbin/mkfs.ext4}
fsck=${E2FSCK:-/opt/homebrew/opt/e2fsprogs/sbin/e2fsck}
sources=$(find vendor/lwext4/src -name '*.c' ! -name ext4_mkfs.c ! -name ext4_mbr.c ! -name ext4.c)
cc -O2 -std=gnu11 -DCONFIG_USE_DEFAULT_CFG=1 -DCONFIG_HAVE_OWN_ERRNO=1 -DCONFIG_DEBUG_PRINTF=0 -DCONFIG_DEBUG_ASSERT=0 -DCONFIG_BLOCK_DEV_CACHE_SIZE=32 -DCONFIG_MAX_TRUNCATE_SIZE=1048576 -Ivendor/lwext4/include $sources c/adapter.c tests/adapter-test.c -o "$d/adapter-test"
"$mkfs" -q -F -b 4096 -O ^orphan_file,^metadata_csum_seed "$d/base.img" 64M
cp "$d/base.img" "$d/rename-base.img"
"$d/adapter-test" "$d/rename-base.img" rename-prepare
for cut in 0 1 2 3 4 8 12 20 40 80; do
    cp "$d/rename-base.img" "$d/rename.img"
    status=0
    "$d/adapter-test" "$d/rename.img" rename-cut "$cut" || status=$?
    [ "$status" = 77 ]
    cp "$d/rename.img" "$d/independent-replay.img"
    E2FSCK="$fsck" python3 tests/check-recovery.py "$d/independent-replay.img"
    "$d/adapter-test" "$d/rename.img" rename-verify
    "$fsck" -fn "$d/rename.img"
done
cp "$d/base.img" "$d/posix.img"
"$d/adapter-test" "$d/posix.img" posix
"$fsck" -fn "$d/posix.img"
cp "$d/base.img" "$d/orphan.img"
status=0
"$d/adapter-test" "$d/orphan.img" orphancrash || status=$?
[ "$status" = 77 ]
cp "$d/orphan.img" "$d/independent-replay.img"
E2FSCK="$fsck" python3 tests/check-recovery.py "$d/independent-replay.img"
"$d/adapter-test" "$d/orphan.img" recover
"$fsck" -fn "$d/orphan.img"
cp "$d/base.img" "$d/rw.img"
"$d/adapter-test" "$d/rw.img" write
"$d/adapter-test" "$d/rw.img" verify
"$fsck" -fn "$d/rw.img"
"$d/adapter-test" "$d/rw.img" readonly
cp "$d/base.img" "$d/error.img"
"$d/adapter-test" "$d/error.img" ioerror
cp "$d/base.img" "$d/flusherror.img"
"$d/adapter-test" "$d/flusherror.img" flusherror
for cut in 0 3 12 40; do
    cp "$d/base.img" "$d/crash.img"
    status=0
    "$d/adapter-test" "$d/crash.img" crash "$cut" || status=$?
    [ "$status" = 77 ]
    cp "$d/crash.img" "$d/independent-replay.img"
    E2FSCK="$fsck" python3 tests/check-recovery.py "$d/independent-replay.img"
    "$d/adapter-test" "$d/crash.img" recover
    "$fsck" -fn "$d/crash.img"
done

cp "$d/base.img" "$d/unsupported.img"
python3 - "$d/unsupported.img" <<'PYCODE'
import struct,sys
with open(sys.argv[1], 'r+b') as f:
    f.seek(1024+92)
    flags=struct.unpack('<I',f.read(4))[0]
    f.seek(1024+92)
    f.write(struct.pack('<I',flags|0x1000))
PYCODE
"$d/adapter-test" "$d/unsupported.img" reject
cp "$d/base.img" "$d/large.img"
debugfs=${DEBUGFS:-/opt/homebrew/opt/e2fsprogs/sbin/debugfs}
"$debugfs" -w -R 'write /dev/null /large' "$d/large.img" >/dev/null 2>&1
"$debugfs" -w -R 'set_inode_field /large size 5368709243' "$d/large.img" >/dev/null 2>&1
"$d/adapter-test" "$d/large.img" large
"$fsck" -fn "$d/large.img"

# The metadata seed must survive a UUID change. Indexed directories and an
# external extent tree exercise more than the superblock/inode paths.
mkdir -p "$d/seed-source/many"
python3 - "$d" <<'PYSEED'
from pathlib import Path
import sys
p=Path(sys.argv[1])
for i in range(300):(p/'seed-source/many'/f'file-{i}').write_bytes(b'seed fixture')
(p/'seed-source/extents').touch()
(p/'seed-debugfs.cmd').write_text(''.join(f'fallocate /extents {i*2} {i*2}\n' for i in range(20))+'set_inode_field /extents size 163840\n')
PYSEED
"$mkfs" -q -F -b 4096 -O orphan_file,metadata_csum_seed -d "$d/seed-source" "$d/seed.img" 64M
"$debugfs" -w -f "$d/seed-debugfs.cmd" "$d/seed.img" >/dev/null 2>&1
"$fsck" -fyD "$d/seed.img"
tune2fs=${TUNE2FS:-/opt/homebrew/opt/e2fsprogs/sbin/tune2fs}
"$tune2fs" -U random "$d/seed.img"
cc -O2 -std=gnu11 -DCONFIG_USE_DEFAULT_CFG=1 -DCONFIG_HAVE_OWN_ERRNO=1 -DCONFIG_DEBUG_PRINTF=1 -DCONFIG_DEBUG_ASSERT=0 -DCONFIG_BLOCK_DEV_CACHE_SIZE=32 -DCONFIG_MAX_TRUNCATE_SIZE=1048576 -Ivendor/lwext4/include $sources c/adapter.c tests/adapter-test.c -o "$d/adapter-seed-test"
python3 - "$d" <<'PYCHECK'
from pathlib import Path
import subprocess,hashlib,sys
p=Path(sys.argv[1]);image=p/'seed.img';before=hashlib.sha256(image.read_bytes()).digest()
r=subprocess.run([str(p/'adapter-seed-test'),str(image),'seed'],text=True,capture_output=True)
(p/'seed-checksums.log').write_text(r.stdout+r.stderr)
assert r.returncode==0,r.stderr
assert 'checksum failed' not in r.stdout.lower(), 'see seed-checksums.log'
assert before==hashlib.sha256(image.read_bytes()).digest(), 'read-only mount wrote image'
print('stored seed: indexed directory, external unwritten extents, no checksum warnings, image unchanged')
PYCHECK
"$d/adapter-test" "$d/seed.img" reject
"$fsck" -fn "$d/seed.img"

python3 - "$d" <<'PYCORRUPT'
from pathlib import Path
import struct,sys,hashlib,subprocess
p=Path(sys.argv[1]);data=bytearray((p/'seed.img').read_bytes())
def crc32c(data):
    crc=0xffffffff
    for b in data:
        crc ^= b
        for _ in range(8):crc=(crc>>1)^(0x82f63b78 if crc&1 else 0)
    return crc
assert crc32c(data[1024:1024+1020])==struct.unpack_from('<I',data,1024+1020)[0]
seed=struct.unpack_from('<I',data,1024+0x270)[0]
struct.pack_into('<I',data,1024+0x270,seed^0x12345678)
struct.pack_into('<I',data,1024+1020,crc32c(data[1024:1024+1020]))
image=p/'seed-corrupt.img';image.write_bytes(data);before=hashlib.sha256(data).digest()
subprocess.run([str(p/'adapter-test'),str(image),'integrity'],check=True)
assert before==hashlib.sha256(image.read_bytes()).digest(), 'corrupted read-only image was written'
PYCORRUPT

# Last-close reclamation must survive a crash without clean unmount, and a
# later writable service mount must not be allowed to repair the test first.
for block in 1024 2048 4096; do
    "$mkfs" -q -F -b "$block" -O ^orphan_file,^metadata_csum_seed "$d/reap-$block.img" 64M
    for round in 1 2 3 4; do
        for phase in reap-crash verify-crash; do
            status=0
            "$d/adapter-test" "$d/reap-$block.img" "$phase" || status=$?
            [ "$status" = 77 ]
            cp "$d/reap-$block.img" "$d/independent-replay.img"
            E2FSCK="$fsck" python3 tests/check-recovery.py "$d/independent-replay.img"
        done
    done
    for cut in 0 1 2 3 4 8 12 20 40 80; do
        cp "$d/reap-$block.img" "$d/reap-cut.img"
        status=0
        "$d/adapter-test" "$d/reap-cut.img" reap-cut "$cut" || status=$?
        [ "$status" = 77 ]
        E2FSCK="$fsck" python3 tests/check-recovery.py "$d/reap-cut.img"
    done
    "$mkfs" -q -F -b "$block" -O ^orphan_file,^metadata_csum_seed "$d/orphan-$block.img" 64M
    status=0
    "$d/adapter-test" "$d/orphan-$block.img" orphancrash || status=$?
    [ "$status" = 77 ]
    cp "$d/orphan-$block.img" "$d/independent-replay.img"
    E2FSCK="$fsck" python3 tests/check-recovery.py "$d/independent-replay.img"
    status=0
    "$d/adapter-test" "$d/orphan-$block.img" verify-crash || status=$?
    [ "$status" = 77 ]
    cp "$d/orphan-$block.img" "$d/independent-replay.img"
    E2FSCK="$fsck" python3 tests/check-recovery.py "$d/independent-replay.img"
    echo "last-close crash/restart summaries and native/Linux orphan recovery: $block-byte blocks passed"
done
