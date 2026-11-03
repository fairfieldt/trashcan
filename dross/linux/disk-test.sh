#!/bin/sh
# Boots dross (unix kernel) in QEMU with a USB stick holding a GPT disk with one ext4 partition, and checks what
# Linux programs read through the mount: files, a symlink, a 300-entry directory, a 20 MB file's hash, and that
# writes are refused. Needs e2fsprogs (mkfs.ext4 -d) and the staged Unix userland (unix/stage-hw.py) for BusyBox.
set -eu
cd "$(dirname "$0")/.."
mkfs=$(command -v mkfs.ext4 || echo /opt/homebrew/opt/e2fsprogs/sbin/mkfs.ext4)
d=target/disktest
rm -rf "$d"
mkdir -p "$d/tree/test/a/b/c"
echo "hello from ext4" > "$d/tree/test/hello.txt"
ln -s hello.txt "$d/tree/test/link"
i=1
while [ $i -le 300 ]; do echo $i > "$d/tree/test/a/b/c/f$i"; i=$((i + 1)); done
head -c 20000000 /dev/urandom > "$d/tree/test/random.bin"
sum=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$d/tree/test/random.bin")
cat > "$d/tree/test/check.sh" <<EOF
set -e
[ "\$(cat /disk/test/hello.txt)" = "hello from ext4" ]
[ "\$(cat /disk/test/link)" = "hello from ext4" ]
[ "\$(ls /disk/test/a/b/c | wc -l)" = 300 ]
[ "\$(cat /disk/test/a/b/c/f123)" = 123 ]
[ "\$(sha256sum /disk/test/random.bin | cut -c1-64)" = $sum ]
! touch /disk/test/new 2>/dev/null
echo "disk test: ok"
EOF
cp linux/disk-write.sh "$d/tree/test/write.sh"
cp linux/disk-verify.sh "$d/tree/test/verify.sh"
"$mkfs" -q -b 4096 -O ^orphan_file,^metadata_csum_seed -L oscar -d "$d/tree" "$d/fs.img" 64M
# A GPT around it: protective MBR, header, entries, the partition from sector 2048, backup entries and header.
python3 - "$d/fs.img" "$d/disk.img" <<'PY'
import struct, sys, uuid, zlib
fs = open(sys.argv[1], 'rb').read()
S, first = 512, 2048
last = first + len(fs) // S - 1
total = last + 1 + 2048
entry = (uuid.UUID('0FC63DAF-8483-4772-8E79-3D69D8477DE4').bytes_le + uuid.uuid4().bytes_le
         + struct.pack('<QQQ', first, last, 0) + 'oscar'.encode('utf-16-le').ljust(72, b'\0'))
entries = entry.ljust(128 * 128, b'\0')
def header(cur, alt, at):
    h = struct.pack('<8sIIIIQQQQ16sQIII', b'EFI PART', 0x10000, 92, 0, 0, cur, alt, 34, total - 34,
                    uuid.uuid4().bytes_le, at, 128, 128, zlib.crc32(entries))
    return h[:16] + struct.pack('<I', zlib.crc32(h)) + h[20:]
img = bytearray(total * S)
img[446:462] = struct.pack('<BBBBBBBBII', 0, 0, 2, 0, 0xee, 0xff, 0xff, 0xff, 1, total - 1)
img[510:512] = b'\x55\xaa'
img[S:S + 92] = header(1, total - 1, 2)
img[2 * S:2 * S + len(entries)] = entries
img[first * S:first * S + len(fs)] = fs
img[(total - 33) * S:(total - 33) * S + len(entries)] = entries
img[(total - 1) * S:(total - 1) * S + 92] = header(total - 1, 1, total - 33)
open(sys.argv[2], 'wb').write(img)
PY
[ "${DISK_PREPARE_ONLY:-0}" = 1 ] && exit 0
KERNEL_FEATURES=unix INIT_CONF=linux/qemu-disk.conf ./run-qemu.sh \
    -drive if=none,id=stick,format=raw,file="$d/disk.img" -device usb-storage,bus=xhci.0,drive=stick
