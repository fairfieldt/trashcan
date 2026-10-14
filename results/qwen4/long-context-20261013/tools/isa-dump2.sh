#!/bin/bash
# RADV shader statistics and disassembly for a fake Tahiti (RADV_FORCE_FAMILY, null winsys) in the kbuild VM: native
# build of the worktree there (each pipeline preceded by a PIPELINE NAME line, a sed in the VM copy only), then every pipeline is created by `qwen4 vram-probe`. Writes isa/stats-TAG.log and
# isa/shaders-TAG.log here. usage: isa-dump.sh TAG
set -euo pipefail
tag=$1
here=$HOME/dev/claude-work/flashnext-20261012/long
src=trashcan-src-agent-a43eb2477e990f76f
vm="ssh -F $HOME/.lima/kbuild/ssh.config lima-kbuild"
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a43eb2477e990f76f
rsync -a --delete -e "ssh -F $HOME/.lima/kbuild/ssh.config" --exclude target Cargo.toml Cargo.lock engine "lima-kbuild:$src/"
$vm "cd $src && grep -q PIPELINE engine/src/vulkan.rs || sed -i 's/^\( *\)let pipe =$/\1eprintln!(\"PIPELINE {variant}\");\n\1let pipe =/' engine/src/vulkan.rs"
$vm "cd $src && ~/.cargo/bin/cargo build --release 2>&1 | grep -E '^error|Finished'"
$vm "cd $src && RADV_FORCE_FAMILY=tahiti RADV_DEBUG=shaderstats,nocache timeout 120 ./target/release/trashcan qwen4 vram-probe > /dev/null 2> isa-stats.log; RADV_FORCE_FAMILY=tahiti RADV_DEBUG=shaders,nocache timeout 120 ./target/release/trashcan qwen4 vram-probe > /dev/null 2> isa-shaders.log; true"
mkdir -p "$here/isa"
scp -q -F "$HOME/.lima/kbuild/ssh.config" "lima-kbuild:$src/isa-stats.log" "$here/isa/stats-$tag.log"
scp -q -F "$HOME/.lima/kbuild/ssh.config" "lima-kbuild:$src/isa-shaders.log" "$here/isa/shaders-$tag.log"
echo "isa/stats-$tag.log isa/shaders-$tag.log"
