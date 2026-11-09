#!/bin/sh
# Build the hosted server with its native capability lifecycle adapter.
set -eu
cd "$(dirname "$0")/.."
: "${DROSS_MUSL_LIBC:?Set DROSS_MUSL_LIBC to the matching retained x86_64 shared musl loader}"
command -v glslc >/dev/null
command -v zig >/dev/null
export TRASHCAN_DROSS_LIBC="$DROSS_MUSL_LIBC"
export TRASHCAN_DROSS_UNWIND="$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-musl/lib/self-contained/libunwind.a"
export TRASHCAN_BUILD_SHA="${TRASHCAN_BUILD_SHA:-$(git rev-parse HEAD)}"
printf 'SOURCE %s\n' "$TRASHCAN_BUILD_SHA"
python3 - <<'PY'
import os,struct
from pathlib import Path
b=Path(os.environ['TRASHCAN_DROSS_LIBC']).read_bytes()
assert b[:6]==b'\x7fELF\x02\x01' and struct.unpack_from('<H',b,18)[0]==62, 'requires x86_64 ELF musl'
sha=os.environ['TRASHCAN_BUILD_SHA']
assert len(sha)==40 and all(c in '0123456789abcdef' for c in sha), 'requires full source commit'
PY
mkdir -p target/dross-server/tools target/dross-server/bin
cat > target/dross-server/tools/cc <<'CC'
#!/bin/sh
for arg in "$@"; do
    shift
    case "$arg" in -lc|-ldl|-lgcc_s) ;; *) set -- "$@" "$arg" ;; esac
done
exec zig cc -target x86_64-linux-musl -dynamic "$@" "$TRASHCAN_DROSS_UNWIND" "$TRASHCAN_DROSS_LIBC" -Wl,--dynamic-linker=/lib/ld-musl-x86_64.so.1
CC
chmod +x target/dross-server/tools/cc
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER="$PWD/target/dross-server/tools/cc"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_RUSTFLAGS='-C target-feature=-crt-static -C link-self-contained=no'
cargo build --locked --release --target x86_64-unknown-linux-musl -p trashcan --features dross
objcopy="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin/llvm-objcopy"
"$objcopy" --strip-debug target/x86_64-unknown-linux-musl/release/trashcan target/dross-server/bin/trashcan.partial
python3 - <<'PY'
from pathlib import Path
import hashlib,struct
p=Path('target/dross-server/bin/trashcan.partial');b=p.read_bytes()
off=struct.unpack_from('<Q',b,32)[0];size,count=struct.unpack_from('<HH',b,54)
assert any(struct.unpack_from('<I',b,off+i*size)[0]==3 for i in range(count)), 'missing shared musl interpreter'
final=p.with_suffix('');p.replace(final)
print(hashlib.sha256(b).hexdigest(),final)
PY
