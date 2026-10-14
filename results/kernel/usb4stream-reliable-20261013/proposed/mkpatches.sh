#!/bin/bash
# In the kbuild VM: proposed stream.c fixes against linux-7.2.9 with kernel/patches-7.2 applied, as unified diffs.
set -euo pipefail
src=~/kernel/7.2/linux-7.2.9
work=~/stream-proposed
rm -rf "$work" && mkdir -p "$work/a/drivers/thunderbolt" "$work/b/drivers/thunderbolt"
cp "$src/drivers/thunderbolt/stream.c" "$work/a/drivers/thunderbolt/stream.c"
cd "$work"

# 1. Close without waiting for the receive ring: its in-flight list holds every posted buffer, so it never empties.
cp a/drivers/thunderbolt/stream.c b/drivers/thunderbolt/stream.c
python3 - b/drivers/thunderbolt/stream.c <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
old = """	/* Wait for the ring to complete any outstanding frames */
	tb_ring_flush(sdev->tx_ring.ring, 500);
	tb_ring_stop(sdev->tx_ring.ring);
	tb_ring_flush(sdev->rx_ring.ring, 500);
	tb_ring_stop(sdev->rx_ring.ring);
"""
new = """	/*
	 * Wait for the Tx ring to send what it holds (the CLOSE frame). The Rx
	 * ring is not flushed: its posted buffers stay in flight until frames
	 * arrive, so the wait always ran the full timeout.
	 */
	tb_ring_flush(sdev->tx_ring.ring, 500);
	tb_ring_stop(sdev->tx_ring.ring);
	tb_ring_stop(sdev->rx_ring.ring);
"""
assert s.count(old) == 1
open(p, 'w').write(s.replace(old, new))
EOF
diff -u a/drivers/thunderbolt/stream.c b/drivers/thunderbolt/stream.c > 0001-close-without-rx-flush.diff || true

# 2. Damaged frames: give the buffer back instead of returning its bytes, count them (configfs rx_dropped).
cp a/drivers/thunderbolt/stream.c b/drivers/thunderbolt/stream.c
python3 ~/patch2.py b/drivers/thunderbolt/stream.c
diff -u a/drivers/thunderbolt/stream.c b/drivers/thunderbolt/stream.c > 0002-drop-damaged-frames.diff || true
# Both together must compile: an out-of-tree build of the module against the tree (nothing written into it).
mkdir -p mod
cp b/drivers/thunderbolt/stream.c mod/stream.c
patch -s -p1 -d "$work" -o "$work/mod/stream.c" a/drivers/thunderbolt/stream.c < 0002-drop-damaged-frames.diff
patch -s mod/stream.c < <(sed 's#a/drivers/thunderbolt/stream.c#mod/stream.c#; s#b/drivers/thunderbolt/stream.c#mod/stream.c#' 0001-close-without-rx-flush.diff)
printf 'obj-m += thunderbolt_stream_proposed.o\nthunderbolt_stream_proposed-y := stream.o\n' > mod/Kbuild
make -s -C "$src" M="$work/mod" ARCH=x86_64 CROSS_COMPILE=x86_64-linux-gnu- modules 2>&1 | grep -E 'error|warning' || true
ls -la mod/*.ko
wc -l "$work"/*.diff
