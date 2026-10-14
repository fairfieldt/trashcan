#!/bin/bash
# Watch a Mac while a job runs on it: every 5 s an ssh round trip (5 s timeout); prints a line when it stops answering
# for 30 s, and exits when JOBLOG shows the job released.
# usage: alive.sh MACHINE JOBLOG
m=$1 log=$2
down=0
while ! grep -q 'released\|ended while waiting\|request failed' "$log" 2>/dev/null; do
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$m" true 2> /dev/null; then
        [ $down -ge 30 ] && echo "$m answers again at $(date +%H:%M:%S)"
        down=0
    else
        down=$((down + 5))
        [ $down -eq 30 ] && echo "$m NOT ANSWERING for 30 s at $(date +%H:%M:%S)"
    fi
    sleep 5
done
echo "job done: $(tail -1 "$log")"
