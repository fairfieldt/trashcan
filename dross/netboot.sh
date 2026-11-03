#!/bin/sh
# Chooses what a test machine set up for network boot (README, "Network boot") boots next: "dross" chainloads
# target/netboot/dross.efi (build.sh with NETBOOT=1), "linux" leaves GRUB's own default. The machine's GRUB
# reads target/netboot/boot.cfg over HTTP at every boot.
#   ./netboot.sh dross|linux
set -eu
cd "$(dirname "$0")"
mkdir -p target/netboot
case "${1:-}" in
dross)
    cat > target/netboot/boot.cfg <<'CFG'
menuentry "dross (network)" --id dross-net {
    echo "loading dross.efi from ${dross_server}"
    chainloader (http,${dross_server})/dross.efi
}
set default=dross-net
set timeout=1
CFG
    ;;
linux) : > target/netboot/boot.cfg ;;
*) echo "usage: $0 dross|linux" >&2; exit 2 ;;
esac
echo "next boot: $1"
