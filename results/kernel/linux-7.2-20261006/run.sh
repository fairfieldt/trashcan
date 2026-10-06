#!/bin/bash
# Kernel A/B on incite: same pinned engine binary and xgpu on 6.12.111-p2p and 7.2.9-p2p.
# usage (on incite): [TAG=x] run.sh   -> writes into results/kernel/linux-7.2-20261006/$(uname -r)[-x]/
set -euo pipefail
cd ~/trashcan
R=results/kernel/linux-7.2-20261006/$(uname -r)${TAG:+-$TAG}
B=results/engine/prefill-tiled-20261003/bin/candidate
M=~/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf
mkdir -p "$R"
exec 9>"$HOME/trashcan-exp/.lock"; flock 9
{
  uname -a; cat /proc/cmdline
  grep -E "^CONFIG_(PCI_P2PDMA|DMABUF_MOVE_NOTIFY|USB4|USB4_NET|USB4_STREAM)=" /boot/config-$(uname -r)
  sha256sum $B bench/xgpu/xgpu
  for d in 02:00.0 06:00.0; do
    echo "$d $(sudo lspci -s $d -vv | grep -E 'LnkSta:' | tr -s ' ')"
    echo "$d vbios $(cat /sys/bus/pci/devices/0000:$d/vbios_version 2>/dev/null)"
    echo "$d runtime_status $(cat /sys/bus/pci/devices/0000:$d/power/runtime_status)"
  done
} > "$R/system.txt" 2>&1
sudo dmesg | grep -i -E "amdgpu|radeon|vfct|vbios|p2p|thunderbolt" > "$R/dmesg-gpu.txt" || true
sudo dmesg | grep -E "WARNING|BUG|Oops|error" > "$R/dmesg-warn.txt" || true
scripts/clockmon.sh "$R/clocks.txt" & mon=$!
trap "kill $mon 2>/dev/null || true" EXIT
cd bench/xgpu
for i in 1 2 3; do ./xgpu p2p-verify > "$OLDPWD/$R/xgpu-verify-$i.txt" 2>&1; echo "exit $?" >> "$OLDPWD/$R/xgpu-verify-$i.txt"; done
./xgpu p2p-wall > "$OLDPWD/$R/xgpu-p2p-wall.txt" 2>&1; echo "exit $?" >> "$OLDPWD/$R/xgpu-p2p-wall.txt"
cd "$OLDPWD"
for rep in 1 2 3 4; do
  l=pp512-$rep
  sudo -n python3 scripts/latency-run.py $B bench $M --pp 512 --tg 128 --batch 512 --progress --profile $R/$l.tsv > $R/$l.out 2> $R/$l.log
  echo PASS $l; cat $R/$l.out
done
