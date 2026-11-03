#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p target/gpud-musl/tools target/gpud-musl/bin
cat > target/gpud-musl/tools/cc <<'CC'
#!/bin/sh
for a in "$@"; do
    shift
    case "$a" in --target=*) ;; *) set -- "$@" "$a" ;; esac
done
exec zig cc -target x86_64-linux-musl "$@"
CC
printf '#!/bin/sh\nexec zig ar "$@"\n' > target/gpud-musl/tools/ar
chmod +x target/gpud-musl/tools/cc target/gpud-musl/tools/ar
export CC_x86_64_unknown_linux_musl="$PWD/target/gpud-musl/tools/cc"
export AR_x86_64_unknown_linux_musl="$PWD/target/gpud-musl/tools/ar"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=rust-lld
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_RUSTFLAGS='-C link-self-contained=yes'
cargo build --release --target x86_64-unknown-linux-musl --target-dir target/gpud-musl -p gpud
objcopy="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin/llvm-objcopy"
for binary in gpud gpud-dev gpud-devctl; do
    "$objcopy" --strip-all "target/gpud-musl/x86_64-unknown-linux-musl/release/$binary" "target/gpud-musl/bin/$binary.tmp"
    mv "target/gpud-musl/bin/$binary.tmp" "target/gpud-musl/bin/$binary"
    shasum -a 256 "target/gpud-musl/bin/$binary"
done
