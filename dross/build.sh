#!/bin/sh
# Builds the loader and the kernel and lays out an EFI system partition in target/esp:
#   EFI/BOOT/BOOTX64.EFI  the loader
#   dross.elf             the kernel
#   modules/              user programs the kernel starts or hands to init
# PROFILE=release for an optimized build.
set -eu
cd "$(dirname "$0")"
PROFILE=${PROFILE:-debug}
flag=""
[ "$PROFILE" = release ] && flag=--release
cargo build -p dross-boot --target x86_64-unknown-uefi $flag
cargo build -p dross --target x86_64-unknown-none $flag
cargo build -p init -p selftest -p pcid -p edutest -p usbd --target x86_64-unknown-none $flag
esp=target/esp
rm -rf "$esp"
mkdir -p "$esp/EFI/BOOT"
cp "target/x86_64-unknown-uefi/$PROFILE/dross-boot.efi" "$esp/EFI/BOOT/BOOTX64.EFI"
cp "target/x86_64-unknown-none/$PROFILE/dross" "$esp/dross.elf"
mkdir -p "$esp/modules"
for m in init selftest pcid edutest usbd; do
    cp "target/x86_64-unknown-none/$PROFILE/$m" "$esp/modules/$m"
done
# Linux programs and their files, if build-linux.sh made them.
if [ -d target/linux ]; then
    cp target/linux/* "$esp/modules/"
fi
# What init starts (INIT_CONF, e.g. linux/qemu-init.conf for run-qemu.sh).
if [ -n "${INIT_CONF:-}" ]; then
    cp "$INIT_CONF" "$esp/modules/init.conf"
fi
echo "$esp"
