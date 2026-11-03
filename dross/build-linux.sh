#!/bin/sh
# Builds the Linux programs dross runs, as static x86_64 musl executables, into target/linux:
#   posixtest                          checks dross's Linux interface
#   gpud, tbd, ethd                    the drivers (driver/), with gpud-platform's dross backend
#   test-*                             the drivers' unit tests
#   dyntest                            a dynamically linked C program (linux/dyntest)
#   oscar-chat-gui                     the desktop's chat window (desktop/chat)
# and target/linux/rootfs.tar from linux/rootfs (plus /lib/firmware from FIRMWARE, a directory, if set), with
# musl's dynamic linker from Alpine's minirootfs (downloaded once into target/alpine, checked by its SHA-256) and
# dyntest's library.
# Needs the x86_64-unknown-linux-musl Rust target and zig (as the C compiler for gpud's C and dyntest).
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

alpine=alpine-minirootfs-3.24.3-x86_64.tar.gz
mkdir -p target/alpine
if [ ! -f "target/alpine/$alpine" ]; then
    curl -fsSL -o "target/alpine/$alpine.part" "https://dl-cdn.alpinelinux.org/alpine/v3.24/releases/x86_64/$alpine"
    mv "target/alpine/$alpine.part" "target/alpine/$alpine"
fi
python3 -I -c 'import hashlib, sys; sys.exit(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest() != sys.argv[2])' \
    "target/alpine/$alpine" f8e1cafba9b1b283b5d30d06f9885c8d2bc161ffe9083ade8e2e157813a53407 \
    || { echo "checksum mismatch: target/alpine/$alpine" >&2; exit 1; }

stage=$PWD/target/rootfs
rm -rf "$stage"
cp -R linux/rootfs "$stage"
mkdir -p "$stage/usr/bin"
cp "$out/posixtest" "$stage/usr/bin/dross-budget-probe"
tar -xzf "target/alpine/$alpine" -C "$stage" ./lib/ld-musl-x86_64.so.1
# oscar-chat (also as ocsar-chat) for the shell, and the chat window as a module.
(cd desktop && CARGO_PROFILE_RELEASE_STRIP=true CARGO_TARGET_DIR=$PWD/../target/chat cargo build --release --target $musl -p oscar-chat --features gui)
mkdir -p "$stage/usr/bin"
cp "target/chat/$musl/release/oscar-chat" "$stage/usr/bin/"
ln -s oscar-chat "$stage/usr/bin/ocsar-chat"
cp "target/chat/$musl/release/oscar-chat-gui" "$out/"
mkdir -p "$stage/usr/lib"
zig cc -target x86_64-linux-musl -O2 -shared -fPIC linux/dyntest/lib.c -o "$stage/usr/lib/libdrosstest.so"
zig build-exe -target x86_64-linux-musl -dynamic -lc -O ReleaseSmall linux/dyntest/main.c -femit-bin="$out/dyntest"
if [ -n "${FIRMWARE:-}" ]; then
    mkdir -p "$stage/lib/firmware"
    cp -R "$FIRMWARE"/. "$stage/lib/firmware/"
fi
(cd "$stage" && COPYFILE_DISABLE=1 tar --format ustar -cf "$out/rootfs.tar" .)
ls "$out"
