#!/bin/bash
# Receive a tar stream on PORT into ~/models/qwen3.8-flash-next, forwarding it to NEXT (a TB host name) if given.
set -o pipefail
port=$1 next=${2:-}
mkdir -p ~/models/qwen3.8-flash-next && cd ~/models/qwen3.8-flash-next || exit 1
if [ -n "$next" ]; then
  nc -d -l "$port" | tee >(nc -N "$next" "$port" > /dev/null) | tar -xf -
else
  nc -d -l "$port" | tar -xf -
fi
echo "$(hostname) received rc=$?"
