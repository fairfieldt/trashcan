#!/bin/bash
# Builds libgpud_drm.so and drminfo for x86-64 Linux (run where x86_64-linux-gnu-gcc is: the kbuild VM).
set -euo pipefail
cd "$(dirname "$0")"
out=${1:-../target/client}
mkdir -p "$out"
cc=${CC:-x86_64-linux-gnu-gcc}
flags="-O2 -g -Wall -Wextra -Wno-unused-parameter -D__user= -I../linux/include/uapi -iquote ../shim/include"
$cc $flags -fPIC -shared -o "$out/libgpud_drm.so" gpud_drm.c -ldl -lpthread
$cc $flags -o "$out/drminfo" drminfo.c -ldl
echo "built $out/libgpud_drm.so $out/drminfo"
