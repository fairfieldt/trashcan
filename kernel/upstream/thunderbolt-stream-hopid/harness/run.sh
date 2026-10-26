#!/bin/bash
# Compiles the HopID functions of drivers/thunderbolt/stream.c unchanged in userspace against a model of the
# XDomain HopID IDAs (shim.h) and runs main.c's scenarios, in which HopID allocations fail. No hardware.
# usage: run.sh STREAM_C [OUT_DIR]   (prints the scenarios; exits 1 if any HopID is freed that the stream
# doesn't hold or one is left allocated)
set -euo pipefail
src=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
here=$(cd "$(dirname "$0")" && pwd)
work=${2:-$here/build}; mkdir -p "$work"
python3 -I "$here/extract.py" "$src" tbstream_dev_xdomain tbstream_dev_release tbstream_dev_alloc_in_hopid \
	tbstream_dev_alloc_out_hopid tbstream_dev_attach_stream tbstream_dev_detach_stream > "$work/stream_funcs.inc"
cc -std=gnu11 -Wall -Wno-unused-function -I"$here" -I"$work" "$here/main.c" -o "$work/hopid-replay"
"$work/hopid-replay"
