#!/bin/bash
# Build the engine for incite in the kbuild lima VM (aarch64 Debian 13, x86_64 cross toolchain, same glibc
# as incite) and copy the binary to DEST on incite, or on every node in HOSTS. Keeps compile load off the nodes.
# Each checkout builds in its own VM directory, so parallel checkouts do not overwrite each other.
# usage: [BUILD_ONLY=1 | HOSTS="incite invite"] scripts/cross-build.sh [DEST]   (default: trashcan-exp/qwen/target/release/trashcan)
set -euo pipefail
cd "$(dirname "$0")/.."
dest=${1:-trashcan-exp/qwen/target/release/trashcan}
src=trashcan-src-$(basename "$PWD")
ssh_vm="ssh -F $HOME/.lima/kbuild/ssh.config"
rsync -a --delete -e "$ssh_vm" --exclude target Cargo.toml Cargo.lock engine "lima-kbuild:$src/"
$ssh_vm lima-kbuild "cd $src && ~/.cargo/bin/cargo build --release --target x86_64-unknown-linux-gnu 2>&1 | grep -E '^(warning|error)|Finished' ; test \${PIPESTATUS[0]} -eq 0"
mkdir -p target
$ssh_vm lima-kbuild "cat $src/target/x86_64-unknown-linux-gnu/release/trashcan" > target/trashcan-incite
chmod +x target/trashcan-incite
if [ "${BUILD_ONLY:-0}" = 1 ]; then
    echo "built target/trashcan-incite (deployment skipped)"
    exit 0
fi
for host in ${HOSTS:-incite}; do
    ssh "$host" "mkdir -p \$(dirname $dest)" && rsync -a target/trashcan-incite "$host:$dest"
done
echo "built and copied to ${HOSTS:-incite}:$dest"
