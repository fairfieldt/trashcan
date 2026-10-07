#!/bin/bash
# Exit (and so notify) when incite or invite stops answering SSH twice in a row, or netconsole logs a kernel problem.
log=~/dev/claude-work/trashcan-kbuild/netconsole-incite.log
start=$(wc -l < "$log")
miss_incite=0; miss_invite=0
while true; do
  for h in incite invite; do
    if ssh -o ConnectTimeout=5 -o BatchMode=yes $h true 2>/dev/null; then eval miss_$h=0; else eval miss_$h=\$\(\(miss_$h+1\)\); fi
  done
  if [ $miss_incite -ge 2 ] || [ $miss_invite -ge 2 ]; then echo "$(date) unreachable: incite=$miss_incite invite=$miss_invite"; tail -20 "$log"; exit 1; fi
  new=$(tail -n +$((start+1)) "$log" | grep -E "WARNING|BUG|Oops|timeout reading config|activation failed|wedge|hung|panic|failed to add XDomain" | head -5)
  if [ -n "$new" ]; then echo "$(date) netconsole:"; echo "$new"; exit 2; fi
  sleep 10
done
