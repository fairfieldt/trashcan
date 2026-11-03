#!/bin/sh
# Delegated-range write/persistence tests with unchanged outside-image sentinels.
set -eu
cd "$(dirname "$0")/.."
BLOCK_RANGE_TEST=1 exec ./linux/disk-write-test.sh
