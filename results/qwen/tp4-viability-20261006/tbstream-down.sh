#!/bin/bash
# Remove the USB4STREAMs named NAME on node HOST (run from the Mac, on both nodes). thunderbolt_stream stays loaded.
# usage: tbstream-down.sh HOST [NAME]
set -euo pipefail
host=$1 name=${2:-xmac}
ssh "$host" "for d in /sys/kernel/config/thunderbolt/stream/*/$name; do [ -d \"\$d\" ] && sudo rmdir \"\$d\" && echo removed \$d; done; true"
