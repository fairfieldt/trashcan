#!/bin/sh
# Boots dross in QEMU (q35, OVMF) shaped like a Mac Pro 6,1: an Ivy Bridge CPU with 24 threads, an Intel
# IOMMU without interrupt remapping, the kernel log on stdio (and in target/qemu.log), plus QEMU's edu test
# device for the driver path, and a USB keyboard and mouse on an xHCI controller (typed on through the
# monitor). Exits 0 when init reports that every check passed and the IOMMU stopped the edu driver's deliberate
# DMA to an unmapped address.
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
log=target/qemu.log
: > "$log"
# Once usbd has set up the keyboard and mouse, type "ab", move the mouse and click, through QEMU's monitor.
python3 -I - "$log" target/monitor.sock >/dev/null 2>&1 <<'PY' &
import socket, sys, time
log, sock = sys.argv[1], sys.argv[2]
for _ in range(1200):
    time.sleep(0.1)
    try:
        if "usbd: input ready" in open(log, errors="replace").read():
            break
    except OSError:
        pass
else:
    sys.exit(0)
s = socket.socket(socket.AF_UNIX)
s.connect(sock)
for cmd in ["sendkey a", "sendkey b", "mouse_move 20 10", "mouse_button 1", "mouse_button 0"]:
    time.sleep(0.3)
    s.sendall(cmd.encode() + b"\n")
time.sleep(0.5)
PY
set +e
injector=$!
set -o pipefail
timeout "$TIMEOUT" qemu-system-x86_64 \
    -machine q35,i8042=off -accel tcg,thread=multi \
    -cpu IvyBridge,+x2apic \
    -smp "$SMP",sockets=1,threads=$threads -m "$MEM" \
    -device intel-iommu,intremap=off \
    -device edu \
    -device qemu-xhci,id=xhci -device usb-kbd,bus=xhci.0,usb_version=1 -device usb-mouse,bus=xhci.0 \
    -monitor unix:target/monitor.sock,server,nowait \
    -drive if=pflash,format=raw,readonly=on,file="$fw" \
    -drive if=pflash,format=raw,file=target/ovmf-vars.fd \
    -drive format=raw,file=fat:rw:"$esp" \
    -device isa-debug-exit,iobase=0xf4,iosize=4 \
    -serial stdio -display none -no-reboot -net none "$@" | tee "$log"
status=$?
kill $injector 2>/dev/null
set -e
# The edu driver's DMA to an unmapped address must have been stopped by the IOMMU.
if [ $status -eq 33 ] && ! grep -q "iommu: fault: 00:02.0 write at iova" "$log"; then
    echo "qemu: no IOMMU fault for the edu device's unmapped DMA" >&2
    exit 1
fi
# isa-debug-exit: (value << 1) | 1; the kernel writes 0x10 for success.
case $status in
33) echo "qemu: boot ok"; exit 0 ;;
35) echo "qemu: kernel reported failure" >&2; exit 1 ;;
124) echo "qemu: timed out after ${TIMEOUT}s" >&2; exit 1 ;;
*) echo "qemu: exit status $status" >&2; exit 1 ;;
esac
