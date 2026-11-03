#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p target/doom-tools
cat > target/doom-tools/cc <<'CC'
#!/bin/sh
for arg in "$@"; do
    shift
    case "$arg" in --target=*) ;; *) set -- "$@" "$arg" ;; esac
done
exec zig cc -target x86_64-linux-musl "$@"
CC
cat > target/doom-tools/ar <<'AR'
#!/bin/sh
exec zig ar "$@"
AR
chmod +x target/doom-tools/cc target/doom-tools/ar
export CC_x86_64_unknown_linux_musl="$PWD/target/doom-tools/cc"
export AR_x86_64_unknown_linux_musl="$PWD/target/doom-tools/ar"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=rust-lld
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_RUSTFLAGS='-C link-self-contained=yes'
cargo build --release --target x86_64-unknown-linux-musl -p oscar-doom --bins ${DOOM_FEATURES:+--features "$DOOM_FEATURES"}
python3 - <<'PY'
from pathlib import Path
import hashlib,urllib.request,zipfile
out=Path('target/freedoom');out.mkdir(exist_ok=True)
archive=out/'freedoom-0.13.0.zip'
expected='3f9b264f3e3ce503b4fb7f6bdcb1f419d93c7b546f4df3e874dd878db9688f59'
if not archive.exists():
    with urllib.request.urlopen('https://github.com/freedoom/freedoom/releases/download/v0.13.0/freedoom-0.13.0.zip') as response:
        archive.write_bytes(response.read())
if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:raise SystemExit('Freedoom archive checksum mismatch')
with zipfile.ZipFile(archive) as z:
    for name in ['freedoom1.wad','COPYING.txt','CREDITS.txt','CREDITS-MUSIC.txt']:
        (out/name).write_bytes(z.read('freedoom-0.13.0/'+name))
PY
