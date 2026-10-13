#!/bin/bash
# Copy the cross-built binary to ~/trashcan-exp/flashnext/trashcan-moe on every Mac and print each copy's sha256.
set -u
bin=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a7f9db84020875d15/target/trashcan-incite
for h in incite indite inflight alight invite enlight; do
    rsync -a "$bin" "${h}:trashcan-exp/flashnext/trashcan-moe" || echo "$h: copy failed"
done
echo "local $(shasum -a 256 "$bin" | cut -c1-16)"
for h in incite indite inflight alight invite enlight; do
    echo "$h $(ssh "$h" 'sha256sum ~/trashcan-exp/flashnext/trashcan-moe' | cut -c1-16)"
done
