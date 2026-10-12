#!/bin/bash
# Stream the G-iq4 Flash-Next GGUF (G-iq4/ plus the Q8_0 n-gram shard it links to) from incite along the TB ring with
# tar over netcat: incite -> indite -> inflight -> alight and incite -> enlight -> invite. Each relay writes and forwards.
set -uo pipefail
here=$(dirname "$0")
PORT=5077
for h in alight inflight indite invite enlight; do ssh "$h" 'cat > ~/relay-node.sh' < "$here/relay-node.sh"; done
recv() { ssh "$1" bash relay-node.sh "$PORT" "${2:-}" < /dev/null; }
t0=$(date +%s)
recv alight & sleep 1
recv inflight alight.tb0 & sleep 1
recv indite inflight.tb1 & sleep 1
recv invite & sleep 1
recv enlight invite.tb0 & sleep 2
ssh incite "set -o pipefail; cd ~/models/qwen3.8-flash-next && tar -cf - G-iq4 Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf | tee >(nc -N indite.tb0 $PORT > /dev/null) | nc -N enlight.tb1 $PORT > /dev/null; echo incite sent rc=\$?" < /dev/null
wait
echo "all done in $(( $(date +%s) - t0 )) s"
