#!/bin/sh
# Freestanding x86-64 ELF; no libc, Linux syscall layer or host loader.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
dross=$(CDPATH= cd -- "$here/../.." && pwd)
out=${OUT:-"$dross/target/c-api-test"}
mkdir -p "$out"
zig cc -target x86_64-freestanding-none -std=c11 -O2 -g \
    -ffreestanding -fno-builtin -fno-stack-protector -mno-red-zone \
    -fno-pic -fno-pie -nostdlib -Wall -Wextra -Werror \
    -I "$dross/protocol/include" -I "$here" \
    -Wl,-T,"$dross/user/user.ld" -Wl,--build-id=none \
    "$here/start.S" "$here/main.c" -o "$out/c-api-test"
printf '%s\n' "$out/c-api-test"
