#!/bin/bash
# A six-Mac session's summaries in one file: the bench pairs, each run's link transports, MTP, and the clocks.
# usage: session-summary.sh SESSION_DIR TAG > SESSION_DIR/summary.txt
d=$1 tag=$2
here=$(dirname "$0")
echo "# bench runs (tools/chain_sum.py)"
python3 "$here/chain_sum.py" "$d" "$tag"
echo
echo "# link transports (tools/transport.py)"
python3 "$here/transport.py" "$d" "$tag"
if [ -f "$d/$tag-mtp-base.json" ]; then
    echo
    echo "# MTP (tools/mtp_summary.py, tools/mtp_same.py)"
    python3 "$here/mtp_summary.py" "$d/$tag-mtp-base.json" "$d/$tag-mtp-cand.json"
    python3 "$here/mtp_same.py" "$d/$tag-mtp-base.json" "$d/$tag-mtp-cand.json"
fi
echo
echo "# clocks (tools/nodeclocks.py)"
python3 "$here/nodeclocks.py" "$d"
