#!/bin/bash
# Copy a local binary to ~/trashcan-exp/flashnext/NAME on every Mac and print each copy's sha256 prefix.
# usage: deploy2.sh LOCAL_BINARY NAME      (NAME must start with trashcan-moe)
set -u
bin=$1 name=$2
case $name in trashcan-moe*) ;; *) echo "bad name"; exit 1 ;; esac
for h in incite indite inflight alight invite enlight; do
    rsync -a --rsync-path="nice -n 19 ionice -c3 rsync" "$bin" "${h}:trashcan-exp/flashnext/$name" || echo "$h: copy failed"
done
echo "local $(shasum -a 256 "$bin" | cut -c1-16)"
for h in incite indite inflight alight invite enlight; do
    echo "$h $(ssh -- "$h" "sha256sum ~/trashcan-exp/flashnext/$name" | cut -c1-16)"
done
