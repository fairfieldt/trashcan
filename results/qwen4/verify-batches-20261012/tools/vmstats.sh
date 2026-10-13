#!/bin/bash
# Build natively in the kbuild VM and print RADV (null device, Tahiti, Mesa 25.0.7 as on the Macs) shader stats for
# pipelines matching the args. With --isa the disassembly goes to vfy/stats/stats.txt too.
# usage: vmstats.sh [--isa] NAME_SUBSTRINGS...
set -euo pipefail
wt=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a39bc1d57bef46fec
out=/Users/fairfieldt/dev/claude-work/flashnext-20261012/vfy/stats
vm="ssh -F /Users/fairfieldt/.lima/kbuild/ssh.config"
mkdir -p $out
isa=0
if [ "${1:-}" = --isa ]; then isa=1; shift; fi
cd $wt
rsync -a --delete -e "$vm" --exclude target Cargo.toml Cargo.lock engine lima-kbuild:trashcan-src-agent-a39bc1d57bef46fec/
dbg=shaderstats,nocache
if [ $isa = 1 ]; then dbg=shaders,shaderstats,nocache; fi
$vm lima-kbuild "cd trashcan-src-agent-a39bc1d57bef46fec && ~/.cargo/bin/cargo build --release 2>&1 | grep -E '^error|Finished' ; RADV_FORCE_FAMILY=tahiti RADV_DEBUG=$dbg timeout 120 ./target/release/trashcan qwen4 vram-probe > ~/vfy-stats/out.txt 2> ~/vfy-stats/err.txt; true"
scp -q -F /Users/fairfieldt/.lima/kbuild/ssh.config lima-kbuild:vfy-stats/err.txt $out/stats.txt
python3 /Users/fairfieldt/dev/claude-work/flashnext-20261012/vfy/tools/stats.py $wt/engine/src/vulkan.rs $out/stats.txt "$@"
