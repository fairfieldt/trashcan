#!/bin/sh
# Boots dross in QEMU (q35, OVMF) shaped like a Mac Pro 6,1: an Ivy Bridge CPU with 24 threads, an Intel
# IOMMU without interrupt remapping, the kernel log on stdio. Exits 0 when the kernel reports "boot ok".
#   SMP=24 MEM=8G ./run-qemu.sh [extra qemu args]
set -eu
cd "$(dirname "$0")"
SMP=${SMP:-24}
MEM=${MEM:-8G}
TIMEOUT=${TIMEOUT:-120}
./build.sh >&2
esp=target/esp

fw=""
for d in /opt/homebrew/share/qemu /usr/local/share/qemu /usr/share/qemu /usr/share/OVMF; do
    [ -f "$d/edk2-x86_64-code.fd" ] && fw="$d/edk2-x86_64-code.fd" && vars="$d/edk2-i386-vars.fd" && break
    [ -f "$d/OVMF_CODE.fd" ] && fw="$d/OVMF_CODE.fd" && vars="$d/OVMF_VARS.fd" && break
done
[ -n "$fw" ] || { echo "no OVMF firmware found" >&2; exit 1; }
cp "$vars" target/ovmf-vars.fd

# 12 cores x 2 threads, as on the E5-2697 v2.
threads=1
[ "$SMP" -ge 2 ] && [ $((SMP % 2)) -eq 0 ] && threads=2
set +e
timeout "$TIMEOUT" qemu-system-x86_64 \
    -machine q35 -accel tcg,thread=multi \
    -cpu IvyBridge,+x2apic \
    -smp "$SMP",sockets=1,threads=$threads -m "$MEM" \
    -device intel-iommu,intremap=off \
    -drive if=pflash,format=raw,readonly=on,file="$fw" \
    -drive if=pflash,format=raw,file=target/ovmf-vars.fd \
    -drive format=raw,file=fat:rw:"$esp" \
    -device isa-debug-exit,iobase=0xf4,iosize=4 \
    -serial stdio -display none -no-reboot -net none "$@"
status=$?
set -e
# isa-debug-exit: (value << 1) | 1; the kernel writes 0x10 for success.
case $status in
33) echo "qemu: boot ok"; exit 0 ;;
35) echo "qemu: kernel reported failure" >&2; exit 1 ;;
124) echo "qemu: timed out after ${TIMEOUT}s" >&2; exit 1 ;;
*) echo "qemu: exit status $status" >&2; exit 1 ;;
esac
