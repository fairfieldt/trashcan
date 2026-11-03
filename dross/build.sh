#!/bin/sh
# Builds the loader and the kernel and lays out an EFI system partition in target/esp:
#   EFI/BOOT/BOOTX64.EFI  the loader
#   dross.elf             the kernel
#   modules/              user programs the kernel starts or hands to init (and Oscar's desktop)
# PROFILE=release for an optimized build.
set -eu
cd "$(dirname "$0")"
PROFILE=${PROFILE:-debug}
flag=""
[ "$PROFILE" = release ] && flag=--release
cargo build -p dross-boot --target x86_64-unknown-uefi $flag
# KERNEL_FEATURES=unix builds the kernel with the Unix subsystem (unix/README.md).
cargo build -p dross --target x86_64-unknown-none $flag ${KERNEL_FEATURES:+--features dross/$KERNEL_FEATURES}
cargo build -p init -p selftest -p pcid -p edutest -p usbd -p fsd -p dross-serviced --target x86_64-unknown-none $flag
# This native C fixture checks the public protocol without a Rust runtime.
OUT="$PWD/target/c-api-test" ./user/c-api-test/build.sh >&2
# Oscar's desktop (desktop/README.md): its own workspace.
cargo build --manifest-path desktop/Cargo.toml -p oscar-desktop-native --features native --bins --target x86_64-unknown-none $flag
esp=target/esp
rm -rf "$esp"
mkdir -p "$esp/EFI/BOOT"
cp "target/x86_64-unknown-uefi/$PROFILE/dross-boot.efi" "$esp/EFI/BOOT/BOOTX64.EFI"
cp "target/x86_64-unknown-none/$PROFILE/dross" "$esp/dross.elf"
mkdir -p "$esp/modules"
for m in init selftest pcid edutest usbd fsd serviced; do
    cp "target/x86_64-unknown-none/$PROFILE/$m" "$esp/modules/$m"
done
cp target/c-api-test/c-api-test "$esp/modules/c-api-test"
for m in oscar-session oscar-windowd oscar-terminal trash; do
    cp "desktop/target/x86_64-unknown-none/$PROFILE/$m" "$esp/modules/$m"
done
# Linux programs and their files, if build-linux.sh made them.
if [ -d target/linux ]; then
    cp target/linux/* "$esp/modules/"
fi
# What init starts (INIT_CONF, e.g. linux/qemu-init.conf for run-qemu.sh).
if [ -n "${INIT_CONF:-}" ]; then
    cp "$INIT_CONF" "$esp/modules/init.conf"
fi
# NETBOOT=1: also target/netboot/dross.efi, the loader with the kernel and modules (stripped) built in, for
# booting over the network (README, "Network boot").
if [ -n "${NETBOOT:-}" ]; then
    objcopy=$(ls "$(rustc --print sysroot)"/lib/rustlib/*/bin/llvm-objcopy | head -1)
    rm -rf target/netboot/files
    mkdir -p target/netboot/files
    cp "$esp/dross.elf" "$esp"/modules/* target/netboot/files/
    for f in target/netboot/files/*; do
        if [ "$(head -c 4 "$f" | od -An -tx1 | tr -d ' ')" = 7f454c46 ]; then
            "$objcopy" --strip-all "$f"
        fi
    done
    python3 - target/netboot/files target/netboot/bundle <<'PY'
import os, struct, sys
src, out = sys.argv[1:]
with open(out, "wb") as f:
    f.write(b"DROSSBND")
    for name in sorted(os.listdir(src)):
        data = open(os.path.join(src, name), "rb").read()
        f.write(struct.pack("<H", len(name)) + name.encode() + struct.pack("<Q", len(data)) + data)
    f.write(struct.pack("<H", 0))
PY
    DROSS_BUNDLE="$PWD/target/netboot/bundle" cargo build -p dross-boot --target x86_64-unknown-uefi $flag --features bundle \
        --target-dir target/netboot/cargo
    cp "target/netboot/cargo/x86_64-unknown-uefi/$PROFILE/dross-boot.efi" target/netboot/dross.efi
    ls -l target/netboot/dross.efi
fi
echo "$esp"
