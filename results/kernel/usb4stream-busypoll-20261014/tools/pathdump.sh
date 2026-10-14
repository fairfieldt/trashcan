#!/bin/bash
# On a Mac Pro, as root (sudo -n bash pathdump.sh), read only: tblink's links, the stream groups' attributes, the
# Thunderbolt lines of the kernel log, and from debugfs every host router's registers and, per port, its adapter
# registers and its path table (each in HopID's two path config dwords). Output lines:
#   ROUTER PORT path OFFSET REL IN_HOPID VALUE        ROUTER PORT regs OFFSET REL CAP VSEC VALUE
#   ROUTER - swregs OFFSET REL CAP VSEC VALUE        # comments (host, kernel, links, groups, kernel log)
echo "# host $(hostname) $(uname -r) $(date +%T.%N)"
/usr/local/sbin/tblink links | sed 's/^/# links /'
for g in /sys/kernel/config/thunderbolt/stream/*/*; do
    [ -d "$g" ] || continue
    a=""
    for f in "$g"/*; do
        [ -f "$f" ] && a="$a $(basename "$f")=$(cat "$f" 2> /dev/null | tr -d '\n')"
    done
    echo "# group $g$a"
done
dmesg | grep -iE 'thunderbolt|tbnet|tbstream|xdomain' | tail -n 60 | sed 's/^/# kernel /'
for r in /sys/kernel/debug/thunderbolt/[0-9]*-0; do
    [ -d "$r" ] || continue
    rn=$(basename "$r")
    awk -v r="$rn" '!/^#/ {print r, "-", "swregs", $0}' "$r/regs" 2> /dev/null
    for p in "$r"/port*; do
        [ -d "$p" ] || continue
        pn=$(basename "$p")
        awk -v r="$rn" -v p="$pn" '!/^#/ {print r, p, "regs", $0}' "$p/regs" 2> /dev/null
        awk -v r="$rn" -v p="$pn" '!/^#/ {print r, p, "path", $0}' "$p/path" 2> /dev/null
    done
done
echo "# end $(date +%T.%N)"
