#!/bin/sh
# Builds the Linux programs dross runs, as static x86_64 musl executables, into target/linux:
#   posixtest                          checks dross's Linux interface
#   gpud, tbd, ethd                    the drivers (driver/), with gpud-platform's dross backend
#   test-*                             the drivers' unit tests
# and target/linux/rootfs.tar from linux/rootfs (plus /lib/firmware from FIRMWARE, a directory, if set).
# Needs the x86_64-unknown-linux-musl Rust target and zig (as the C compiler for gpud's C).
set -eu
cd "$(dirname "$0")"
out=$PWD/target/linux
mkdir -p "$out"
tools=$PWD/target/linux-tools
mkdir -p "$tools"
cat > "$tools/cc" <<'SH'
#!/bin/sh
# zig cc for x86_64 musl; drops the --target the cc crate adds (zig spells targets differently).
for a in "$@"; do
    shift
    case "$a" in --target=*) ;; *) set -- "$@" "$a" ;; esac
done
exec zig cc -target x86_64-linux-musl "$@"
SH
printf '#!/bin/sh\nexec zig ar "$@"\n' > "$tools/ar"
chmod +x "$tools/cc" "$tools/ar"
export CC_x86_64_unknown_linux_musl="$tools/cc" AR_x86_64_unknown_linux_musl="$tools/ar"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=rust-lld
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_RUSTFLAGS="-C link-self-contained=yes"
musl=x86_64-unknown-linux-musl

(cd linux/posixtest && CARGO_TARGET_DIR=$PWD/../../target/posixtest cargo build --release --target $musl)
cp "target/posixtest/$musl/release/posixtest" "$out/"

if [ -z "${NO_DRIVERS:-}" ]; then
    (cd ../driver && CARGO_TARGET_DIR=$PWD/../dross/target/driver cargo build --release --target $musl -p gpud -p tbd -p ethd)
    for b in gpud tbd ethd; do
        cp "target/driver/$musl/release/$b" "$out/"
    done
    # Unit tests: the last line of cargo's JSON names each test binary.
    (cd ../driver && CARGO_TARGET_DIR=$PWD/../dross/target/driver cargo test --release --target $musl -p gpud-platform -p tbd -p ethd -p gpud -p tbd-client --no-run --message-format=json) \
        | python3 -I -c '
import json, shutil, sys
out = sys.argv[1]
for line in sys.stdin:
    m = json.loads(line)
    if m.get("reason") == "compiler-artifact" and m.get("profile", {}).get("test") and m.get("executable"):
        shutil.copy(m["executable"], out + "/test-" + m["target"]["name"])
' "$out"
fi

stage=$PWD/target/rootfs
rm -rf "$stage"
cp -R linux/rootfs "$stage"
if [ -n "${FIRMWARE:-}" ]; then
    mkdir -p "$stage/lib/firmware"
    cp -R "$FIRMWARE"/. "$stage/lib/firmware/"
fi
(cd "$stage" && COPYFILE_DISABLE=1 tar --format ustar -cf "$out/rootfs.tar" .)
ls "$out"
