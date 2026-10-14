#!/bin/bash
# Build the engine's unit tests for x86_64 in the kbuild VM (as scripts/cross-build.sh builds the engine) and copy the
# test binary here, to run the x86-only paths (SSSE3 and AVX drafter kernels) on incite.
set -euo pipefail
W=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a173c86cfe38f66c7
cd "$W"
src=trashcan-src-$(basename "$PWD")
ssh_vm="ssh -F $HOME/.lima/kbuild/ssh.config"
rsync -a --delete -e "$ssh_vm" --exclude target Cargo.toml Cargo.lock engine "lima-kbuild:$src/"
bin=$($ssh_vm lima-kbuild "cd $src && ~/.cargo/bin/cargo test --release --target x86_64-unknown-linux-gnu --no-run --message-format=json 2>/dev/null" \
    | python3 -c "import json,sys
for l in sys.stdin:
    try: m = json.loads(l)
    except Exception: continue
    if m.get('reason') == 'compiler-artifact' and m.get('executable') and m['profile'].get('test'): print(m['executable'])" | tail -1)
echo "test binary: $bin"
$ssh_vm lima-kbuild "cat $bin" > /Users/fairfieldt/dev/claude-work/flashnext-20261012/mtp2/prof/trashcan-mtp2-tests
chmod +x /Users/fairfieldt/dev/claude-work/flashnext-20261012/mtp2/prof/trashcan-mtp2-tests
ls -la /Users/fairfieldt/dev/claude-work/flashnext-20261012/mtp2/prof/trashcan-mtp2-tests
