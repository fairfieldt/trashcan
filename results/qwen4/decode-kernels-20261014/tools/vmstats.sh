#!/bin/bash
# Build natively in the kbuild VM and dump RADV (null device, Tahiti, Mesa 25.0.7 as on the Macs) shader stats and
# programs for every pipeline (RADV_DEBUG=shaders,shaderstats,nocache), into $DK/stats/TAG.txt; then print the stats
# of pipelines matching NAME_SUBSTRINGS. The checkout is the one this script is in.
# usage: [DK=DIR] vmstats.sh TAG [NAME_SUBSTRINGS...]   (DK default ~/dev/claude-work/flashnext-20261012/dk2)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
wt=$(cd "$here/../../../.." && pwd)
dk=${DK:-$HOME/dev/claude-work/flashnext-20261012/dk2}
vm="ssh -F $HOME/.lima/kbuild/ssh.config"
src=trashcan-src-$(basename "$wt")
tag=$1; shift
mkdir -p "$dk/stats"
cd "$wt"
rsync -a --delete -e "$vm" --exclude target Cargo.toml Cargo.lock engine lima-kbuild:$src/
$vm lima-kbuild "mkdir -p ~/dk2-stats; cd $src && ~/.cargo/bin/cargo build --release 2>&1 | grep -E '^error|Finished' ; RADV_FORCE_FAMILY=tahiti RADV_DEBUG=shaders,shaderstats,nocache timeout 300 ./target/release/trashcan qwen4 vram-probe > ~/dk2-stats/out.txt 2> ~/dk2-stats/err.txt; true"
scp -q -F "$HOME/.lima/kbuild/ssh.config" lima-kbuild:dk2-stats/err.txt "$dk/stats/$tag.txt"
python3 "$here/stats.py" "$wt/engine/src/vulkan.rs" "$dk/stats/$tag.txt" "$@"
