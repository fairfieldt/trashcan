#!/bin/bash
# Build the engine for incite in the kbuild lima VM (aarch64 Debian 13, x86_64 cross toolchain, same glibc
# as incite) and copy the binary to DEST on incite. Keeps compile load off incite.
# usage: scripts/cross-build.sh [DEST]   (default: trashcan-exp/qwen/target/release/trashcan)
set -euo pipefail
cd "$(dirname "$0")/.."
dest=${1:-trashcan-exp/qwen/target/release/trashcan}
ssh_vm="ssh -F $HOME/.lima/kbuild/ssh.config"
rsync -a --delete -e "$ssh_vm" --exclude target Cargo.toml Cargo.lock engine lima-kbuild:trashcan-src/
$ssh_vm lima-kbuild 'cd trashcan-src && ~/.cargo/bin/cargo build --release --target x86_64-unknown-linux-gnu 2>&1 | grep -E "^(warning|error)|Finished" ; test ${PIPESTATUS[0]} -eq 0'
$ssh_vm lima-kbuild 'cat trashcan-src/target/x86_64-unknown-linux-gnu/release/trashcan' > target/trashcan-incite
chmod +x target/trashcan-incite
ssh incite "mkdir -p \$(dirname $dest)" && rsync -a target/trashcan-incite "incite:$dest"
echo "built and copied to incite:$dest"
