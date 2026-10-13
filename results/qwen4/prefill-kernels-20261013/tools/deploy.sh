#!/bin/bash
# Copy the worktree's cross-built binary to every Mac as ~/trashcan-exp/flashnext/NAME and print the hashes.
# usage: deploy.sh NAME
set -u
name=$1
bin=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a2c014fa1f29b2375/target/trashcan-incite
echo "local $(shasum -a 256 $bin | cut -c1-16)"
for h in incite indite inflight alight invite enlight; do
    rsync -a "$bin" "${h}:trashcan-exp/flashnext/${name}" || echo "FAILED ${h}"
    echo "${h} $(ssh "${h}" "sha256sum ~/trashcan-exp/flashnext/${name}" | cut -c1-16)"
done
