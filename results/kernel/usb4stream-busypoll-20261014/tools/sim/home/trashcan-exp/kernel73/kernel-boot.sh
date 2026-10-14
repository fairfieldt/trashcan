#!/bin/bash
# Stub of kernel-boot.sh on a simulated Mac: `once KREL` makes KREL the next boot's kernel.
set -euo pipefail
mode=$1 krel=$2
[ "$mode" = once ] || [ "$mode" = default ] || { echo "mode must be once or default" >&2; exit 1; }
echo "$krel" > "$SIMHOME/$SIMHOST.next"
echo 'GRUB_DEFAULT="gnulinux-advanced-x>gnulinux-7.3.0-rc7-p2p-advanced-x"'
echo "installed $krel; running $(uname -r)"
