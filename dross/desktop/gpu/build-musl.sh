#!/bin/sh
# Rust std plus an actual shared musl libc: Zig's bundled -lc otherwise stays
# static even with -dynamic, making dlopen report 'Dynamic loading not supported'.
set -eu
cd "$(dirname "$0")/.."
export OSCAR_MUSL_SHARED="${DROSS_MUSL_LIBC:-$PWD/../target/rootfs/lib/ld-musl-x86_64.so.1}"
export OSCAR_MUSL_UNWIND="$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-musl/lib/self-contained/libunwind.a"
[ -f "$OSCAR_MUSL_SHARED" ] || { echo 'build dross Linux rootfs first, or set DROSS_MUSL_LIBC' >&2; exit 1; }
mkdir -p target/gpu-musl/tools target/gpu-musl/bin
cat > target/gpu-musl/tools/cc <<'CC'
#!/bin/sh
for arg in "$@"; do
    shift
    case "$arg" in -lc|-ldl|-lgcc_s) ;; *) set -- "$@" "$arg" ;; esac
done
exec zig cc -target x86_64-linux-musl -dynamic "$@" "$OSCAR_MUSL_UNWIND" "$OSCAR_MUSL_SHARED" -Wl,--dynamic-linker=/lib/ld-musl-x86_64.so.1
CC
chmod +x target/gpu-musl/tools/cc
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER="$PWD/target/gpu-musl/tools/cc"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_RUSTFLAGS='-C target-feature=-crt-static -C link-self-contained=no'
cargo build --release --target x86_64-unknown-linux-musl --target-dir target/gpu-musl -p oscar-desktop-gpu --bins
objcopy="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin/llvm-objcopy"
for binary in oscar-vkcheck oscar-vkdraw oscar-vkshare oscar-windowd-gpu; do
    "$objcopy" --strip-all "target/gpu-musl/x86_64-unknown-linux-musl/release/$binary" "target/gpu-musl/bin/$binary.tmp"
done
python3 - <<'PY'
from pathlib import Path
import hashlib,struct
for name in ['oscar-vkcheck','oscar-vkdraw','oscar-vkshare','oscar-windowd-gpu']:
    f=Path('target/gpu-musl/bin')/(name+'.tmp')
    b=f.read_bytes();off=struct.unpack_from('<Q',b,32)[0];size,count=struct.unpack_from('<HH',b,54)
    assert any(struct.unpack_from('<I',b,off+i*size)[0]==3 for i in range(count)),f'{f}: missing PT_INTERP'
    final=f.with_suffix('')
    f.replace(final)
    print(hashlib.sha256(b).hexdigest(),final)
PY
