#!/bin/sh
# Builds the loader and the kernel and lays out an EFI system partition in target/esp:
#   EFI/BOOT/BOOTX64.EFI  the loader
#   dross.elf             the kernel
# PROFILE=release for an optimized build.
set -eu
cd "$(dirname "$0")"
PROFILE=${PROFILE:-debug}
flag=""
[ "$PROFILE" = release ] && flag=--release
cargo build -p dross-boot --target x86_64-unknown-uefi $flag
cargo build -p dross --target x86_64-unknown-none $flag
esp=target/esp
rm -rf "$esp"
mkdir -p "$esp/EFI/BOOT"
cp "target/x86_64-unknown-uefi/$PROFILE/dross-boot.efi" "$esp/EFI/BOOT/BOOTX64.EFI"
cp "target/x86_64-unknown-none/$PROFILE/dross" "$esp/dross.elf"
echo "$esp"
